"""
Context Management for AgentPress Threads.

This module handles token counting and thread summarization to prevent
reaching the context window limitations of LLM models.
"""

import json
from typing import List, Dict, Any, Optional

from litellm import token_counter, completion, completion_cost, get_model_info
from litellm.utils import trim_messages
from services.supabase import DBConnection
from services.llm import make_llm_api_call
from utils.logger import logger

# Constants for token management
DEFAULT_TOKEN_THRESHOLD = 120000  # 80k tokens threshold for summarization
SUMMARY_TARGET_TOKENS = 10000    # Target ~10k tokens for the summary message
RESERVE_TOKENS = 5000            # Reserve tokens for new messages

class ContextManager:
    """Manages thread context including token counting and summarization."""
    
    def __init__(self, token_threshold: int = DEFAULT_TOKEN_THRESHOLD):
        """Initialize the ContextManager.
        
        Args:
            token_threshold: Token count threshold to trigger summarization
        """
        self.db = DBConnection()
        self.token_threshold = token_threshold
    
    async def get_thread_token_count(self, thread_id: str, model: str = "gpt-4") -> int:
        """Get the current token count for a thread using LiteLLM.
        
        Args:
            thread_id: ID of the thread to analyze
            model: The name of the LLM model being used (for accurate token counting)
            
        Returns:
            The total token count for relevant messages in the thread
        """
        logger.debug(f"Getting token count for thread {thread_id}")
        
        try:
            # Get messages for the thread
            messages = await self.get_messages_for_summarization(thread_id)
            
            if not messages:
                logger.debug(f"No messages found for thread {thread_id}")
                return 0
            
            # Use litellm's token_counter for accurate model-specific counting
            # This is much more accurate than the SQL-based estimation
            token_count = token_counter(model=model, messages=messages)
            
            logger.info(f"Thread {thread_id} has {token_count} tokens (calculated with litellm using model '{model}')")
            return token_count
                
        except Exception as e:
            logger.error(f"Error getting token count: {str(e)}")
            return 0
    
    async def get_messages_for_summarization(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get all LLM messages from the thread that need to be summarized.
        
        This gets messages after the most recent summary or all messages if
        no summary exists. Unlike get_llm_messages, this includes ALL messages 
        since the last summary, even if we're generating a new summary.
        
        Args:
            thread_id: ID of the thread to get messages from
            
        Returns:
            List of message objects to summarize
        """
        logger.debug(f"Getting messages for summarization for thread {thread_id}")
        client = await self.db.client
        
        try:
            # Find the most recent summary message
            summary_result = await client.table('messages').select('created_at') \
                .eq('thread_id', thread_id) \
                .eq('type', 'summary') \
                .eq('is_llm_message', True) \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()
            
            # Get messages after the most recent summary or all messages if no summary
            if summary_result.data and len(summary_result.data) > 0:
                last_summary_time = summary_result.data[0]['created_at']
                logger.debug(f"Found last summary at {last_summary_time}")
                
                # Get all messages after the summary, but NOT including the summary itself
                messages_result = await client.table('messages').select('*') \
                    .eq('thread_id', thread_id) \
                    .eq('is_llm_message', True) \
                    .gt('created_at', last_summary_time) \
                    .order('created_at') \
                    .execute()
            else:
                logger.debug("No previous summary found, getting all messages")
                # Get all messages
                messages_result = await client.table('messages').select('*') \
                    .eq('thread_id', thread_id) \
                    .eq('is_llm_message', True) \
                    .order('created_at') \
                    .execute()
            
            # Parse the message content if needed
            messages = []
            for msg in messages_result.data:
                # Skip existing summary messages - we don't want to summarize summaries
                if msg.get('type') == 'summary':
                    logger.debug(f"Skipping summary message from {msg.get('created_at')}")
                    continue
                    
                # Parse content if it's a string
                content = msg['content']
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass  # Keep as string if not valid JSON
                
                # Ensure we have the proper format for the LLM
                if 'role' not in content and 'type' in msg:
                    # Convert message type to role if needed
                    role = msg['type']
                    if role == 'assistant' or role == 'user' or role == 'system' or role == 'tool':
                        content = {'role': role, 'content': content}
                
                messages.append(content)
            
            logger.info(f"Got {len(messages)} messages to summarize for thread {thread_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages for summarization: {str(e)}", exc_info=True)
            return []
    
    async def create_summary(
        self, 
        thread_id: str, 
        messages: List[Dict[str, Any]], 
        model: str = "gpt-4o-mini"
    ) -> Optional[Dict[str, Any]]:
        """Generate a summary of conversation messages.
        
        Args:
            thread_id: ID of the thread to summarize
            messages: Messages to summarize
            model: LLM model to use for summarization
            
        Returns:
            Summary message object or None if summarization failed
        """
        if not messages:
            logger.warning("No messages to summarize")
            return None
        
        logger.info(f"Creating summary for thread {thread_id} with {len(messages)} messages using model {model}")
        
        # --- Trim messages BEFORE creating the prompt for the summarization model ---    
        try:
            summary_model_info = get_model_info(model)
            summary_max_context = summary_model_info.get('max_input_tokens')
            
            if summary_max_context:
                # Leave a buffer for the system prompt instructions AND the requested completion tokens
                summary_buffer = 5000 # Generous buffer for summary instructions
                # Subtract summary target tokens as well, as it counts towards the total context for some models
                summary_target_tokens = summary_max_context - summary_buffer - SUMMARY_TARGET_TOKENS 
                
                if summary_target_tokens <= 0:
                    logger.error(f"[Summary Trim] Calculated target tokens ({summary_target_tokens}) is too low after subtracting buffer and completion target. Aborting summary pre-trim.")
                    messages_to_summarize = messages # Use original messages
                else:
                    messages_to_summarize_tokens = token_counter(model=model, messages=messages)
                    logger.info(f"[Summary Trim] Tokens in messages to summarize: {messages_to_summarize_tokens}. Target for summarizer input: {summary_target_tokens} (Context: {summary_max_context}, Buffer: {summary_buffer}, Completion: {SUMMARY_TARGET_TOKENS})")

                    if messages_to_summarize_tokens > summary_target_tokens:
                        logger.warning(f"[Summary Trim] Trimming messages before sending to summarization model ({messages_to_summarize_tokens} > {summary_target_tokens})" )
                        # We don't need to preserve specific messages for the summary input, just trim from the start
                        trimmed_summary_input = trim_messages(
                            messages=messages, 
                            model=model, 
                            max_tokens=summary_target_tokens
                        )
                        trimmed_count = token_counter(model=model, messages=trimmed_summary_input)
                        logger.info(f"[Summary Trim] Trimmed messages for summary input down to {trimmed_count} tokens.")
                        messages_to_summarize = trimmed_summary_input # Use trimmed messages
                    else:
                        logger.info("[Summary Trim] No need to trim messages for summarization input.")
                        messages_to_summarize = messages # Use original messages
            else:
                logger.warning(f"[Summary Trim] Could not get max context for summarization model {model}, proceeding without pre-trimming.")
                messages_to_summarize = messages # Use original messages
        except Exception as trim_exc:
            logger.error(f"[Summary Trim] Error during pre-trimming for summarization: {trim_exc}", exc_info=True)
            messages_to_summarize = messages # Fallback to original messages
        # --- End Trim messages --- 

        # Create system message with summarization instructions, using the potentially trimmed messages
        system_message = {
            "role": "system",
            "content": f"""You are a specialized summarization assistant. Your task is to create a concise but comprehensive summary of the conversation history provided below.

THE CONVERSATION HISTORY TO SUMMARIZE IS AS FOLLOWS:
===============================================================
==================== CONVERSATION HISTORY ====================
{messages_to_summarize} 
==================== END OF CONVERSATION HISTORY ====================
===============================================================

Please create the summary now, ensuring it includes all key decisions, conclusions, tool usage, and important context to allow the conversation to continue smoothly. Present it as a narrated list of key points under relevant section headers.
"""
        }
        
        try:
            # Call LLM to generate summary
            response = await make_llm_api_call(
                model_name=model,
                messages=[system_message, {"role": "user", "content": "PLEASE PROVIDE THE SUMMARY NOW."}],
                temperature=0,
                max_tokens=SUMMARY_TARGET_TOKENS,
                stream=False
            )
            
            if response and hasattr(response, 'choices') and response.choices:
                summary_content = response.choices[0].message.content
                
                # Track token usage
                try:
                    token_count = token_counter(model=model, messages=[{"role": "user", "content": summary_content}])
                    cost = completion_cost(model=model, prompt="", completion=summary_content)
                    logger.info(f"Summary generated with {token_count} tokens at cost ${cost:.6f}")
                except Exception as e:
                    logger.error(f"Error calculating token usage: {str(e)}")
                
                # Format the summary message with clear beginning and end markers
                formatted_summary = f"""
======== CONVERSATION HISTORY SUMMARY ========

{summary_content}

======== END OF SUMMARY ========

The above is a summary of the conversation history. The conversation continues below.
"""
                
                # Format the summary message
                summary_message = {
                    "role": "user",
                    "content": formatted_summary
                }
                
                return summary_message
            else:
                logger.error("Failed to generate summary: Invalid response")
                return None
                
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}", exc_info=True)
            return None
        
    async def check_and_summarize_if_needed(
        self, 
        thread_id: str, 
        add_message_callback,
        main_model: str,
        summarization_model: str = "gpt-4o-mini",
        force: bool = False
    ) -> bool:
        """Check if thread needs summarization and summarize if so.
        
        Args:
            thread_id: ID of the thread to check
            add_message_callback: Callback to add the summary message to the thread
            main_model: The LLM model used by the main thread (for token threshold check)
            summarization_model: LLM model to use for generating the summary
            force: Whether to force summarization regardless of token count
            
        Returns:
            True if summarization was performed, False otherwise
        """
        try:
            # Get token count using LiteLLM, using the main model name for accuracy
            token_count = await self.get_thread_token_count(thread_id, model=main_model) 
            
            # If token count is below threshold and not forcing, no summarization needed
            if token_count < self.token_threshold and not force:
                logger.debug(f"Thread {thread_id} has {token_count} tokens, below threshold {self.token_threshold}")
                return False
            
            # Log reason for summarization
            if force:
                logger.info(f"Forced summarization of thread {thread_id} with {token_count} tokens (counted with {main_model})")
            else:
                logger.info(f"Thread {thread_id} exceeds token threshold ({token_count} >= {self.token_threshold}) using model {main_model}, summarizing...")
            
            # Get messages to summarize
            messages = await self.get_messages_for_summarization(thread_id)
            
            # If there are too few messages, don't summarize
            if len(messages) < 3:
                logger.info(f"Thread {thread_id} has too few messages ({len(messages)}) to summarize")
                return False
            
            # Create summary using the specified summarization model
            summary = await self.create_summary(thread_id, messages, model=summarization_model)
            
            if summary:
                # Add summary message to thread
                await add_message_callback(
                    thread_id=thread_id,
                    type="summary",
                    content=summary,
                    is_llm_message=True,
                    metadata={"token_count": token_count}
                )
                
                logger.info(f"Successfully added summary to thread {thread_id}")
                return True
            else:
                logger.error(f"Failed to create summary for thread {thread_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error in check_and_summarize_if_needed: {str(e)}", exc_info=True)
            return False 