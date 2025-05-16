'use client';

import React, { useState, useRef, useEffect, forwardRef, useImperativeHandle } from 'react';
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Send, Square, Loader2, X, Paperclip, Settings, ChevronDown } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { toast } from "sonner";
import { AnimatePresence, motion } from "framer-motion";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { cn } from "@/lib/utils";

// Define API_URL
const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || '';

// Local storage keys
const STORAGE_KEY_MODEL = 'suna-preferred-model';
const DEFAULT_MODEL_ID = "sonnet-3.7"; // Define default model ID

// Define the structure for pending files
interface PendingFileItem {
  file: File;
  desiredName: string; // This will be the relative path for folder uploads
}

interface ChatInputProps {
  onSubmit: (message: string, options?: { model_name?: string; enable_thinking?: boolean }) => void;
  placeholder?: string;
  loading?: boolean;
  disabled?: boolean;
  isAgentRunning?: boolean;
  onStopAgent?: () => void;
  autoFocus?: boolean;
  value?: string;
  onChange?: (value: string) => void;
  onFileBrowse?: () => void; // Kept for potential individual file browsing if needed separately
  sandboxId?: string;
  hideAttachments?: boolean;
}

interface UploadedFile {
  name: string; // This will display the relative path for folder files
  path: string; // Sandbox path, e.g., /workspace/folder/file.txt
  size: number;
}

// Define interface for the ref
export interface ChatInputHandles {
  getPendingFiles: () => PendingFileItem[];
  clearPendingFiles: () => void;
}

export const ChatInput = forwardRef<ChatInputHandles, ChatInputProps>(({
  onSubmit,
  placeholder = "Describe what you need help with...",
  loading = false,
  disabled = false,
  isAgentRunning = false,
  onStopAgent,
  autoFocus = true,
  value: controlledValue,
  onChange: controlledOnChange,
  onFileBrowse,
  sandboxId,
  hideAttachments = false
}, ref) => {
  const isControlled = controlledValue !== undefined && controlledOnChange !== undefined;
  
  const [uncontrolledValue, setUncontrolledValue] = useState('');
  const value = isControlled ? controlledValue : uncontrolledValue;

  // Define model options array earlier so it can be used in useEffect
  const modelOptions = [
    { id: "sonnet-3.7", label: "Sonnet 3.7" },
    { id: "sonnet-3.7-thinking", label: "Sonnet 3.7 (Thinking)" },
    { id: "gpt-4.1", label: "GPT-4.1" },
    { id: "gemini-flash-2.5", label: "Gemini Flash 2.5" }
  ];

  // Initialize state with the default model
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL_ID);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [pendingFiles, setPendingFiles] = useState<PendingFileItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  
  // Expose methods through the ref
  useImperativeHandle(ref, () => ({
    getPendingFiles: () => pendingFiles,
    clearPendingFiles: () => {
      setPendingFiles([]);
      setUploadedFiles([]); // Also clear UI representation of these pending files
    }
  }));

  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const savedModel = localStorage.getItem(STORAGE_KEY_MODEL);
        // Check if the saved model exists and is one of the valid options
        if (savedModel && modelOptions.some(option => option.id === savedModel)) {
          setSelectedModel(savedModel);
        } else if (savedModel) {
          // If invalid model found in storage, clear it
          localStorage.removeItem(STORAGE_KEY_MODEL);
          console.log(`Removed invalid model '${savedModel}' from localStorage. Using default: ${DEFAULT_MODEL_ID}`);
        }
      } catch (error) {
        console.warn('Failed to load preferences from localStorage:', error);
      }
    }
  }, []);
  
  useEffect(() => {
    if (autoFocus && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [autoFocus]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const adjustHeight = () => {
      textarea.style.height = 'auto';
      const newHeight = Math.min(Math.max(textarea.scrollHeight, 24), 200);
      textarea.style.height = `${newHeight}px`;
    };

    adjustHeight();
    
    adjustHeight();

    window.addEventListener('resize', adjustHeight);
    return () => window.removeEventListener('resize', adjustHeight);
  }, [value]);

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY_MODEL, model);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((!value.trim() && uploadedFiles.length === 0) || loading || (disabled && !isAgentRunning)) return;
    
    if (isAgentRunning && onStopAgent) {
      onStopAgent();
      return;
    }
    
    let message = value;
    
    if (uploadedFiles.length > 0) {
      const fileInfo = uploadedFiles.map(file => 
        `[Uploaded File: ${file.name}]`
      ).join('\n');
      message = message ? `${message}\n\n${fileInfo}` : fileInfo;
    }
    
    let baseModelName = selectedModel;
    let thinkingEnabled = false;
    if (selectedModel.endsWith("-thinking")) {
      baseModelName = selectedModel.replace(/-thinking$/, "");
      thinkingEnabled = true;
    }
    
    onSubmit(message, {
      model_name: baseModelName,
      enable_thinking: thinkingEnabled
    });
    
    if (!isControlled) {
      setUncontrolledValue("");
    }
    
    setUploadedFiles([]);
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    if (isControlled) {
      controlledOnChange(newValue);
    } else {
      setUncontrolledValue(newValue);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if ((value.trim() || uploadedFiles.length > 0) && !loading && (!disabled || isAgentRunning)) {
        handleSubmit(e as React.FormEvent);
      }
    }
  };

  const handleFileUpload = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(false);
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(false);
    
    if (!e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
    
    const files = Array.from(e.dataTransfer.files);
    
    handleFileSelection(files);
  };

  const processFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files || event.target.files.length === 0) return;
    
    const files = Array.from(event.target.files);
    handleFileSelection(files);
        
    event.target.value = '';
  };

  // Unified function to handle selected files (both from input and drag-drop)
  const handleFileSelection = (files: File[]) => {
    if (sandboxId) {
      // If we have a sandboxId, upload files directly using their original names
      // The folder upload mechanism is separate and will use pendingFiles for initial submission
      uploadFiles(files.map(f => ({ file: f, desiredName: f.name })));
    } else {
      // Otherwise, store files locally for pending submission
      const filteredFiles = files.filter(file => {
        if (file.size > 50 * 1024 * 1024) {
          toast.error(`File size exceeds 50MB limit: ${file.name}`);
          return false;
        }
        return true;
      });
      
      const newPendingItems: PendingFileItem[] = filteredFiles.map(file => ({
        file,
        desiredName: file.name // For individual files, desiredName is file.name
      }));
      setPendingFiles(prev => [...prev, ...newPendingItems]);
      
      const newUiUploadedFiles: UploadedFile[] = newPendingItems.map(item => ({
        name: item.desiredName,
        path: `/workspace/${item.desiredName}`, // For display purposes
        size: item.file.size
      }));
      setUploadedFiles(prev => [...prev, ...newUiUploadedFiles]);
      filteredFiles.forEach(file => {
        toast.success(`File attached: ${file.name} (pending submission)`);
      });
    }
  };

  const handleDirectorySelection = async () => {
    if (!(window as any).showDirectoryPicker) {
      toast.error("Your browser does not support folder selection. Try a modern browser like Chrome or Edge.");
      return;
    }
    try {
      const directoryHandle = await (window as any).showDirectoryPicker();
      const filesToProcess: { file: File, relativePath: string }[] = [];
      
      async function processDirectory(dirHandle: any, currentPath: string) {
        for await (const entry of dirHandle.values()) {
          const entryPath = currentPath ? `${currentPath}/${entry.name}` : entry.name;
          if (entry.kind === 'file') {
            const file = await entry.getFile();
            if (file.size > 50 * 1024 * 1024) {
              toast.error(`File size exceeds 50MB limit: ${entryPath}`);
              continue;
            }
            // Basic exclusion for known unwanted files/folders (can be expanded)
            const commonExclusions = ['.DS_Store', '.Trashes', 'Thumbs.db'];
            if (commonExclusions.includes(entry.name)) {
              logger.debug(`Skipping excluded file: ${entryPath}`);
              continue;
            }
            filesToProcess.push({ file, relativePath: entryPath });
          } else if (entry.kind === 'directory') {
            const commonDirExclusions = ['.git', '.hg', '.svn', 'node_modules', '__pycache__', 'build', 'dist'];
            if (commonDirExclusions.includes(entry.name)) {
              logger.debug(`Skipping excluded directory: ${entryPath}`);
              continue;
            }
            await processDirectory(entry, entryPath);
          }
        }
      }

      await processDirectory(directoryHandle, '');

      if (filesToProcess.length === 0) {
        toast.info("No files found in the selected directory or all files were filtered.");
        return;
      }

      // Folder uploads are always for pending submission, even if sandboxId exists,
      // because they are part of the initial prompt/context setting.
      const newPendingItems: PendingFileItem[] = filesToProcess.map(item => ({
        file: item.file,
        desiredName: item.relativePath
      }));
      setPendingFiles(prev => [...prev, ...newPendingItems]);

      const newUiUploadedFiles: UploadedFile[] = newPendingItems.map(item => ({
        name: item.desiredName, // Display relative path
        path: `/workspace/${item.desiredName}`, // For display consistency
        size: item.file.size
      }));
      setUploadedFiles(prev => [...prev, ...newUiUploadedFiles]);
      
      toast.success(`Added ${filesToProcess.length} files from '${directoryHandle.name}' for pending submission.`);

    } catch (err: any) {
      if (err.name === 'AbortError') {
        toast.info("Folder selection cancelled.");
      } else {
        console.error("Error selecting directory:", err);
        toast.error(`Could not access the directory: ${err.message}`);
      }
    }
  };

  // This function uploads files *directly* to an existing sandbox (e.g., if sandboxId is known)
  // It's typically used for ad-hoc uploads after a session has started, not for initial pending files.
  // For folder uploads, we always use the pendingFiles mechanism.
  const uploadFiles = async (filesToUpload: PendingFileItem[]) => {
    if (!sandboxId) {
      toast.error("Cannot upload files: Sandbox ID is missing.");
      return;
    }
    try {
      setIsUploading(true);
      const newSuccessfullyUploadedForUI: UploadedFile[] = [];
      
      for (const item of filesToUpload) {
        if (item.file.size > 50 * 1024 * 1024) {
          toast.error(`File size exceeds 50MB limit: ${item.desiredName}`);
          continue;
        }
        
        const formData = new FormData();
        // IMPORTANT: When uploading directly to sandbox, the backend /sandboxes/{id}/files endpoint
        // might expect the filename in a 'path' field or use the file's actual name.
        // The current implementation of that endpoint is not shown, but it likely creates files
        // directly in /workspace, not nested. For this direct upload, we use item.desiredName (original filename for individual files).
        // If this endpoint needs to support relative paths too, it would need similar logic to /agent/initiate.
        // For now, assuming it takes the base filename.
        formData.append('file', item.file, item.file.name); // Send with original file name
        formData.append('path', `/workspace/${item.file.name}`); // Target path in sandbox
                
        const supabase = createClient();
        const { data: { session } } = await supabase.auth.getSession();
        
        if (!session?.access_token) {
          throw new Error('No access token available');
        }
        
        const response = await fetch(`${API_URL}/api/sandboxes/${sandboxId}/files`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${session.access_token}`,
          },
          body: formData
        });
        
        if (!response.ok) {
          throw new Error(`Upload failed: ${response.statusText}`);
        }
        
        newSuccessfullyUploadedForUI.push({
          name: item.desiredName,
          path: `/workspace/${item.file.name}`,
          size: item.file.size
        });
        
        toast.success(`File uploaded: ${item.desiredName}`);
      }
      
      // Add to uploadedFiles for UI, but these are *not* pending submission anymore.
      setUploadedFiles(prev => [...prev, ...newSuccessfullyUploadedForUI]); 
      
    } catch (error) {
      console.error("File upload failed:", error);
      toast.error(typeof error === 'string' ? error : (error instanceof Error ? error.message : "Failed to upload file"));
    } finally {
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const removeUploadedFile = (index: number) => {
    const removedUiFile = uploadedFiles[index];
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
    
    // Also remove from pendingFiles if the removed UI file corresponds to a pending file
    setPendingFiles(prevPending => prevPending.filter(pf => pf.desiredName !== removedUiFile.name));
  };

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-4">
      <AnimatePresence>
        {uploadedFiles.length > 0 && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-2 overflow-hidden"
          >
            <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
              {uploadedFiles.map((file, index) => (
                <motion.div 
                  key={`${file.name}-${index}`}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ duration: 0.15 }}
                  className={cn(
                    "px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded-md flex items-center gap-1.5 group text-sm",
                    !sandboxId ? "border border-blue-200 dark:border-blue-800" : ""
                  )}
                >
                  <span className="truncate max-w-[120px] text-gray-700 dark:text-gray-300">{file.name}</span>
                  <span className="text-xs text-gray-500 dark:text-gray-400 flex-shrink-0">
                    ({formatFileSize(file.size)})
                    {!sandboxId && <span className="ml-1 text-blue-500">(pending)</span>}
                  </span>
                  <Button 
                    type="button" 
                    variant="ghost" 
                    size="icon" 
                    className="h-4 w-4 rounded-full p-0 hover:bg-gray-200 dark:hover:bg-gray-700"
                    onClick={() => {
                      removeUploadedFile(index);
                    }}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div 
        className={cn(
          "flex items-end w-full rounded-lg border border-gray-200 dark:border-gray-900 bg-white dark:bg-black px-3 py-2 shadow-sm transition-all duration-200",
          isDraggingOver ? "border-blue-200 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-950/10" : ""
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="relative flex-1 flex items-center overflow-hidden dark:bg-black">
          <Textarea
            ref={textareaRef}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className={cn(
              "min-h-[24px] max-h-[200px] py-0 px-0 text-sm resize-none border-0 shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 bg-transparent w-full dark:bg-black",
              isDraggingOver ? "opacity-40" : ""
            )}
            disabled={loading || (disabled && !isAgentRunning) || isUploading}
            rows={1}
          />
        </div>
        
        <div className="flex items-center gap-2 pl-2 flex-shrink-0">
          {!isAgentRunning && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button 
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                      >
                        <Settings className="h-4 w-4" />
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="sm:max-w-[360px] p-0 gap-0 border border-border shadow-lg">
                      <DialogHeader className="px-4 pt-4 pb-3 border-b">
                        <DialogTitle className="text-sm font-medium">Select Model</DialogTitle>
                      </DialogHeader>
                      <div className="p-4">
                        <RadioGroup 
                          defaultValue={selectedModel} 
                          onValueChange={handleModelChange}
                          className="grid gap-2"
                        >
                          {modelOptions.map(option => (
                            <div key={option.id} className="flex items-center space-x-2 rounded-md px-3 py-2 cursor-pointer hover:bg-accent">
                              <RadioGroupItem value={option.id} id={option.id} />
                              <Label htmlFor={option.id} className="flex-1 cursor-pointer text-sm font-normal">
                                {option.label}
                              </Label>
                              {selectedModel === option.id && (
                                <span className="text-xs text-muted-foreground">Active</span>
                              )}
                            </div>
                          ))}
                        </RadioGroup>
                      </div>
                    </DialogContent>
                  </Dialog>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p>Settings</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          
          {!hideAttachments && (
            <>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button 
                    type="button"
                    onClick={handleFileUpload}
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                    disabled={loading || (disabled && !isAgentRunning) || isUploading}
                  >
                    <Paperclip className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p>Attach files</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button 
                    type="button"
                    onClick={handleDirectorySelection}
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                    disabled={loading || (disabled && !isAgentRunning) || isUploading}
                  >
                    {isUploading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-folder-up">
                        <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/>
                        <path d="M12 10v6"/>
                        <path d="m15 13-3-3-3 3"/>
                      </svg>
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <p>Add folder</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            </>
          )}
          
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            onChange={processFileUpload}
            multiple
          />
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button 
                  type="submit"
                  onClick={isAgentRunning ? onStopAgent : handleSubmit}
                  variant="ghost"
                  size="icon"
                  className={cn(
                    "h-8 w-8 rounded-md",
                    isAgentRunning 
                      ? "text-red-500 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/30" 
                      : "text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800",
                    ((!value.trim() && uploadedFiles.length === 0) && !isAgentRunning) || loading || (disabled && !isAgentRunning) 
                      ? "opacity-50" 
                      : ""
                  )}
                  disabled={((!value.trim() && uploadedFiles.length === 0) && !isAgentRunning) || loading || (disabled && !isAgentRunning)}
                >
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : isAgentRunning ? (
                    <Square className="h-4 w-4" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="top">
                <p>{isAgentRunning ? 'Stop agent' : 'Send message'}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      {isAgentRunning && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-2 w-full flex items-center justify-center"
        >
          <div className="text-xs text-muted-foreground flex items-center gap-2">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span>Kortix Suna is working...</span>
          </div>
        </motion.div>
      )}
    </div>
  );
});

// Set display name for the component
ChatInput.displayName = 'ChatInput'; 

// Add logger utility if not already present globally, or import it
const logger = {
  debug: console.debug,
  info: console.info,
  warn: console.warn,
  error: console.error,
}; 