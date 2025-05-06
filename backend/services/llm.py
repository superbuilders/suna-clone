"""
LLM API interface for making calls to various language models.

This module provides a unified interface for making API calls to different LLM providers
(OpenAI, Anthropic, Groq, etc.) using LiteLLM. It includes support for:
- Streaming responses
- Tool calls and function calling
- Retry logic with exponential backoff
- Model-specific configurations
- Comprehensive error handling and logging
"""

from typing import Union, Dict, Any, Optional, AsyncGenerator, List
import os
import json
import asyncio
from openai import OpenAIError
import litellm
from litellm import token_counter
from utils.logger import logger
from utils.config import config
from datetime import datetime
import traceback

# litellm.set_verbose=True
litellm.modify_params=True

# Constants
MAX_RETRIES = 1
RATE_LIMIT_DELAY = 30
RETRY_DELAY = 5

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class LLMRetryError(LLMError):
    """Exception raised when retries are exhausted."""
    pass

def setup_api_keys() -> None:
    """Set up API keys from environment variables."""
    providers = ['OPENAI', 'ANTHROPIC', 'GROQ', 'OPENROUTER']
    for provider in providers:
        key = getattr(config, f'{provider}_API_KEY')
        if key:
            logger.debug(f"API key set for provider: {provider}")
        else:
            logger.warning(f"No API key found for provider: {provider}")
    
    # Set up OpenRouter API base if not already set
    if config.OPENROUTER_API_KEY and config.OPENROUTER_API_BASE:
        os.environ['OPENROUTER_API_BASE'] = config.OPENROUTER_API_BASE
        logger.debug(f"Set OPENROUTER_API_BASE to {config.OPENROUTER_API_BASE}")
    
    # Set up AWS Bedrock credentials
    aws_access_key = config.AWS_ACCESS_KEY_ID
    aws_secret_key = config.AWS_SECRET_ACCESS_KEY
    aws_region = config.AWS_REGION_NAME
    
    if aws_access_key and aws_secret_key and aws_region:
        logger.debug(f"AWS credentials set for Bedrock in region: {aws_region}")
        # Configure LiteLLM to use AWS credentials
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        os.environ['AWS_REGION_NAME'] = aws_region
    else:
        logger.warning(f"Missing AWS credentials for Bedrock integration - access_key: {bool(aws_access_key)}, secret_key: {bool(aws_secret_key)}, region: {aws_region}")

async def handle_error(error: Exception, attempt: int, max_attempts: int) -> None:
    """Handle API errors with appropriate delays and logging."""
    delay = RATE_LIMIT_DELAY if isinstance(error, litellm.exceptions.RateLimitError) else RETRY_DELAY
    logger.warning(f"Error on attempt {attempt + 1}/{max_attempts}: {str(error)}")
    logger.debug(f"Waiting {delay} seconds before retry...")
    await asyncio.sleep(delay)

def prepare_params(
    messages: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low'
) -> Dict[str, Any]:
    """Prepare parameters for the API call."""
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,
        "top_p": top_p,
        "stream": stream,
    }

    if api_key:
        params["api_key"] = api_key
    if api_base:
        params["api_base"] = api_base
    if model_id:
        params["model_id"] = model_id

    # Handle token limits
    if max_tokens is not None:
        # For Claude 3.7 in Bedrock, do not set max_tokens or max_tokens_to_sample
        # as it causes errors with inference profiles
        if model_name.startswith("bedrock/") and "claude-3-7" in model_name:
            logger.debug(f"Skipping max_tokens for Claude 3.7 model: {model_name}")
            # Do not add any max_tokens parameter for Claude 3.7
        else:
            param_name = "max_completion_tokens" if 'o1' in model_name else "max_tokens"
            params[param_name] = max_tokens

    # Add tools if provided
    if tools:
        params.update({
            "tools": tools,
            "tool_choice": tool_choice
        })
        logger.debug(f"Added {len(tools)} tools to API parameters")

    # # Add Claude-specific headers
    if "claude" in model_name.lower() or "anthropic" in model_name.lower():
        params["extra_headers"] = {
            # "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"
            "anthropic-beta": "output-128k-2025-02-19"
        }
        logger.debug("Added Claude-specific headers")
    
    # Add OpenRouter-specific parameters
    if model_name.startswith("openrouter/"):
        logger.debug(f"Preparing OpenRouter parameters for model: {model_name}")
        
        # Add optional site URL and app name from config
        site_url = config.OR_SITE_URL
        app_name = config.OR_APP_NAME
        if site_url or app_name:
            extra_headers = params.get("extra_headers", {})
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            if app_name:
                extra_headers["X-Title"] = app_name
            params["extra_headers"] = extra_headers
            logger.debug(f"Added OpenRouter site URL and app name to headers")
    
    # Add Bedrock-specific parameters
    if model_name.startswith("bedrock/"):
        logger.debug(f"Preparing AWS Bedrock parameters for model: {model_name}")
        
        if not model_id and "anthropic.claude-3-7-sonnet" in model_name:
            params["model_id"] = "arn:aws:bedrock:us-west-2:935064898258:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            logger.debug(f"Auto-set model_id for Claude 3.7 Sonnet: {params['model_id']}")

    # Apply Anthropic prompt caching (minimal implementation)
    # Check model name *after* potential modifications (like adding bedrock/ prefix)
    effective_model_name = params.get("model", model_name)
    if "claude" in effective_model_name.lower() or "anthropic" in effective_model_name.lower():
        # Log the Anthropic key being used (partially)
        anthropic_key_to_log = params.get("api_key") or config.ANTHROPIC_API_KEY
        if anthropic_key_to_log:
            partial_key = f"{anthropic_key_to_log[:5]}...{anthropic_key_to_log[-4:]}"
            logger.info(f"Using Anthropic API Key (partial): {partial_key} for model {effective_model_name}")
        else:
            logger.warning(f"No Anthropic API Key found in params or config for model {effective_model_name}")

        messages = params["messages"] # Direct reference, modification affects params

        # Ensure messages is a list
        if not isinstance(messages, list):
            return params # Return early if messages format is unexpected

        # 1. Process the first message if it's a system prompt with string content
        if messages and messages[0].get("role") == "system":
            content = messages[0].get("content")
            if isinstance(content, str):
                # Wrap the string content in the required list structure
                messages[0]["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(content, list):
                 # If content is already a list, check if the first text block needs cache_control
                 for item in content:
                     if isinstance(item, dict) and item.get("type") == "text":
                         if "cache_control" not in item:
                             item["cache_control"] = {"type": "ephemeral"}
                             break # Apply to the first text block only for system prompt

        # 2. Find and process relevant user and assistant messages
        last_user_idx = -1
        second_last_user_idx = -1
        last_assistant_idx = -1

        for i in range(len(messages) - 1, -1, -1):
            role = messages[i].get("role")
            if role == "user":
                if last_user_idx == -1:
                    last_user_idx = i
                elif second_last_user_idx == -1:
                    second_last_user_idx = i
            elif role == "assistant":
                if last_assistant_idx == -1:
                    last_assistant_idx = i

            # Stop searching if we've found all needed messages
            if last_user_idx != -1 and second_last_user_idx != -1 and last_assistant_idx != -1:
                 break

        # Helper function to apply cache control
        def apply_cache_control(message_idx: int, message_role: str):
            if message_idx == -1:
                return

            message = messages[message_idx]
            content = message.get("content")

            if isinstance(content, str):
                message["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        if "cache_control" not in item:
                           item["cache_control"] = {"type": "ephemeral"}

        # Apply cache control to the identified messages
        apply_cache_control(last_user_idx, "last user")
        apply_cache_control(second_last_user_idx, "second last user")
        apply_cache_control(last_assistant_idx, "last assistant")

    # Add reasoning_effort for Anthropic models if enabled
    use_thinking = enable_thinking if enable_thinking is not None else False
    is_anthropic = "anthropic" in effective_model_name.lower() or "claude" in effective_model_name.lower()

    if is_anthropic and use_thinking:
        effort_level = reasoning_effort if reasoning_effort else 'low'
        params["reasoning_effort"] = effort_level
        params["temperature"] = 1.0 # Required by Anthropic when reasoning_effort is used
        logger.info(f"Anthropic thinking enabled with reasoning_effort='{effort_level}'")

    return params

async def make_llm_api_call(
    messages: List[Dict[str, Any]],
    model_name: str,
    response_format: Optional[Any] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low'
) -> Union[Dict[str, Any], AsyncGenerator]:
    """
    Make an API call to a language model using LiteLLM.
    
    Args:
        messages: List of message dictionaries for the conversation
        model_name: Name of the model to use (e.g., "gpt-4", "claude-3", "openrouter/openai/gpt-4", "bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
        response_format: Desired format for the response
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in the response
        tools: List of tool definitions for function calling
        tool_choice: How to select tools ("auto" or "none")
        api_key: Override default API key
        api_base: Override default API base URL
        stream: Whether to stream the response
        top_p: Top-p sampling parameter
        model_id: Optional ARN for Bedrock inference profiles
        enable_thinking: Whether to enable thinking
        reasoning_effort: Level of reasoning effort
        
    Returns:
        Union[Dict[str, Any], AsyncGenerator]: API response or stream
        
    Raises:
        LLMRetryError: If API call fails after retries
        LLMError: For other API-related errors
    """
    # debug <timestamp>.json messages 
    logger.debug(f"Making LLM API call to model: {model_name} (Thinking: {enable_thinking}, Effort: {reasoning_effort})")
    params = prepare_params(
        messages=messages,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        api_key=api_key,
        api_base=api_base,
        stream=stream,
        top_p=top_p,
        model_id=model_id,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort
    )
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES}")
            # logger.debug(f"API request parameters: {json.dumps(params, indent=2)}")
            
            # Log final payload size details before the actual call
            final_messages = params.get("messages", [])
            final_message_count = len(final_messages)
            final_token_count_estimate = -1
            try:
                final_token_count_estimate = token_counter(model=params.get("model"), messages=final_messages)
                logger.info(f"[LLM Service] Final check BEFORE litellm.acompletion (attempt {attempt+1}): Model='{params.get('model')}', Messages='{final_message_count}', Estimated Tokens='{final_token_count_estimate}'")
            except Exception as count_err:
                logger.warning(f"[LLM Service] Could not estimate token count before call: {count_err}")
                logger.info(f"[LLM Service] Final check BEFORE litellm.acompletion (attempt {attempt+1}): Model='{params.get('model')}', Messages='{final_message_count}', Estimated Tokens='ERROR'")

            # Dynamically adjust max_tokens for Anthropic models to prevent context overflow
            # This is done after the initial final_token_count_estimate and before the acompletion call.
            if params.get("model") and "anthropic" in params.get("model", "").lower() and final_token_count_estimate > 0:
                try:
                    model_info = litellm.get_model_info(params.get("model"))
                    # For Anthropic, max_input_tokens is the total context window for prompt + completion.
                    actual_max_context = model_info.get('max_input_tokens') 
                    
                    if actual_max_context:
                        # Correction factor based on observed underestimation (e.g., 142k actual vs 116k estimate -> ~1.22)
                        # Using 1.25 as a slightly safer margin.
                        correction_factor = 1.25 
                        inflated_input_estimate = int(final_token_count_estimate * correction_factor)
                        
                        # Tokens available for output generation
                        available_for_output = actual_max_context - inflated_input_estimate
                        
                        # Apply a small safety buffer for the output itself
                        output_safety_buffer = 2000  # e.g., 2k tokens
                        dynamically_calculated_max_tokens = max(1, available_for_output - output_safety_buffer) # Ensure at least 1 token

                        original_requested_max_tokens = params.get("max_tokens") # As initially set by prepare_params

                        new_max_tokens = dynamically_calculated_max_tokens
                        if original_requested_max_tokens is not None:
                            # If an explicit max_tokens was requested, respect it as an upper bound
                            new_max_tokens = min(original_requested_max_tokens, dynamically_calculated_max_tokens)
                        
                        if (original_requested_max_tokens is None) or (new_max_tokens < original_requested_max_tokens) :
                             logger.info(
                                 f"Dynamically adjusting 'max_tokens' for Anthropic model {params.get('model')}. "
                                 f"Original Req: {original_requested_max_tokens}, "
                                 f"Estimated Input (raw): {final_token_count_estimate}, "
                                 f"Estimated Input (inflated): {inflated_input_estimate}, "
                                 f"Model Max Context: {actual_max_context}, "
                                 f"Available for Output (pre-buffer): {available_for_output}, "
                                 f"New 'max_tokens': {new_max_tokens}"
                             )
                        
                        params["max_tokens"] = new_max_tokens

                        # Safety check: if max_tokens becomes non-positive, log an error.
                        # This indicates the inflated input estimate alone might be exceeding the context limit.
                        # The input trimming logic in thread_manager.py should ideally prevent this earlier,
                        # but this is a final safeguard.
                        if params["max_tokens"] <= 0:
                             logger.error(
                                 f"Anthropic 'max_tokens' calculated to <= 0 ({params['max_tokens']}). "
                                 f"Inflated input estimate ({inflated_input_estimate}) "
                                 f"likely exceeds model context ({actual_max_context}). "
                                 f"This will probably cause an API error."
                             )
                             # Forcing to 1 as Anthropic might require at least 1.
                             params["max_tokens"] = 1
                             
                except Exception as adj_err:
                    logger.warning(f"Error dynamically adjusting max_tokens for Anthropic: {adj_err}", exc_info=True)
            
            response = await litellm.acompletion(**params)
            logger.debug(f"Successfully received API response from {model_name}")
            # logger.debug(f"Response: {response}") # Avoid logging potentially large successful responses
            return response
            
        except litellm.exceptions.ContextWindowExceededError as cwe:
            # Log details specifically for context window errors
            logger.error(f"[LLM Service] ContextWindowExceededError on attempt {attempt + 1}/{MAX_RETRIES}. Model: {params.get('model')}, Final Estimated Tokens: {final_token_count_estimate}, Error: {cwe}", exc_info=True)
            last_error = cwe # Store specific error
            # Decide if retry makes sense - for context window, it usually doesn't unless message structure changes
            if attempt + 1 < MAX_RETRIES:
                 logger.warning("Context window exceeded, retrying might not help but proceeding...")
                 await handle_error(cwe, attempt, MAX_RETRIES) # Use standard backoff
            else:
                 logger.error("Context window exceeded, max retries reached.")
                 raise LLMRetryError(f"Failed to make API call after {MAX_RETRIES} attempts. Last error: {cwe}") from cwe

        except (litellm.exceptions.RateLimitError, litellm.InternalServerError, OpenAIError, json.JSONDecodeError) as e:
            last_error = e
            await handle_error(e, attempt, MAX_RETRIES)
            # If MAX_RETRIES is 1, this block will be exited after the first attempt if an error occurs,
            # and the error will be re-raised by the logic outside the loop.
            
        except Exception as e:
            logger.error(f"Unexpected error during API call: {str(e)}", exc_info=True)
            raise LLMError(f"API call failed: {str(e)}")
    
    error_msg = f"Failed to make API call after {MAX_RETRIES} attempts"
    if last_error:
        error_msg += f". Last error: {str(last_error)}"
    logger.error(error_msg, exc_info=True)
    raise LLMRetryError(error_msg)

# Initialize API keys on module import
setup_api_keys()

# Test code for OpenRouter integration
async def test_openrouter():
    """Test the OpenRouter integration with a simple query."""
    test_messages = [
        {"role": "user", "content": "Hello, can you give me a quick test response?"}
    ]
    
    try:
        # Test with standard OpenRouter model
        print("\n--- Testing standard OpenRouter model ---")
        response = await make_llm_api_call(
            model_name="openrouter/openai/gpt-4o-mini",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        
        # Test with deepseek model
        print("\n--- Testing deepseek model ---")
        response = await make_llm_api_call(
            model_name="openrouter/deepseek/deepseek-r1-distill-llama-70b",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        
        # Test with Mistral model
        print("\n--- Testing Mistral model ---")
        response = await make_llm_api_call(
            model_name="openrouter/mistralai/mixtral-8x7b-instruct",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        
        return True
    except Exception as e:
        print(f"Error testing OpenRouter: {str(e)}")
        return False

async def test_bedrock():
    """Test the AWS Bedrock integration with a simple query."""
    test_messages = [
        {"role": "user", "content": "Hello, can you give me a quick test response?"}
    ]
    
    try:    
        response = await make_llm_api_call(
            model_name="bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
            model_id="arn:aws:bedrock:us-west-2:935064898258:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=test_messages,
            temperature=0.7,
            # Claude 3.7 has issues with max_tokens, so omit it
            # max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        
        return True
    except Exception as e:
        print(f"Error testing Bedrock: {str(e)}")
        return False

if __name__ == "__main__":
    import asyncio
        
    test_success = asyncio.run(test_bedrock())
    
    if test_success:
        print("\n✅ integration test completed successfully!")
    else:
        print("\n❌ Bedrock integration test failed!")
