import os
from typing import List, Optional, AsyncGenerator, Tuple
import zipfile
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter, Form, Depends, Request
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel

from utils.logger import logger
from utils.auth_utils import get_current_user_id_from_jwt, get_user_id_from_stream_auth, get_optional_user_id
from sandbox.sandbox import get_or_start_sandbox
from services.supabase import DBConnection
from agent.api import get_or_create_project_sandbox


# Initialize shared resources
router = APIRouter(tags=["sandbox"])
db = None

def initialize(_db: DBConnection):
    """Initialize the sandbox API with resources from the main API."""
    global db
    db = _db
    logger.info("Initialized sandbox API with database connection")

class FileInfo(BaseModel):
    """Model for file information"""
    name: str
    path: str
    is_dir: bool
    size: int
    mod_time: str
    permissions: Optional[str] = None

async def verify_sandbox_access(client, sandbox_id: str, user_id: Optional[str] = None):
    """
    Verify that a user has access to a specific sandbox based on account membership.
    
    Args:
        client: The Supabase client
        sandbox_id: The sandbox ID to check access for
        user_id: The user ID to check permissions for. Can be None for public resource access.
        
    Returns:
        dict: Project data containing sandbox information
        
    Raises:
        HTTPException: If the user doesn't have access to the sandbox or sandbox doesn't exist
    """
    # Find the project that owns this sandbox
    project_result = await client.table('projects').select('*').filter('sandbox->>id', 'eq', sandbox_id).execute()
    
    if not project_result.data or len(project_result.data) == 0:
        raise HTTPException(status_code=404, detail="Sandbox not found")
    
    project_data = project_result.data[0]

    if project_data.get('is_public'):
        return project_data
    
    # For private projects, we must have a user_id
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required for this resource")
    
    account_id = project_data.get('account_id')
    
    # Verify account membership
    if account_id:
        account_user_result = await client.schema('basejump').from_('account_user').select('account_role').eq('user_id', user_id).eq('account_id', account_id).execute()
        if account_user_result.data and len(account_user_result.data) > 0:
            return project_data
    
    raise HTTPException(status_code=403, detail="Not authorized to access this sandbox")

async def get_sandbox_by_id_safely(client, sandbox_id: str):
    """
    Safely retrieve a sandbox object by its ID, using the project that owns it.
    
    Args:
        client: The Supabase client
        sandbox_id: The sandbox ID to retrieve
    
    Returns:
        Sandbox: The sandbox object
        
    Raises:
        HTTPException: If the sandbox doesn't exist or can't be retrieved
    """
    # Find the project that owns this sandbox
    project_result = await client.table('projects').select('project_id').filter('sandbox->>id', 'eq', sandbox_id).execute()
    
    if not project_result.data or len(project_result.data) == 0:
        logger.error(f"No project found for sandbox ID: {sandbox_id}")
        raise HTTPException(status_code=404, detail="Sandbox not found - no project owns this sandbox ID")
    
    project_id = project_result.data[0]['project_id']
    logger.debug(f"Found project {project_id} for sandbox {sandbox_id}")
    
    try:
        # Get the sandbox
        sandbox, retrieved_sandbox_id, sandbox_pass = await get_or_create_project_sandbox(client, project_id)
        
        # Verify we got the right sandbox
        if retrieved_sandbox_id != sandbox_id:
            logger.warning(f"Retrieved sandbox ID {retrieved_sandbox_id} doesn't match requested ID {sandbox_id} for project {project_id}")
            # Fall back to the direct method if IDs don't match (shouldn't happen but just in case)
            sandbox = await get_or_start_sandbox(sandbox_id)
        
        return sandbox
    except Exception as e:
        logger.error(f"Error retrieving sandbox {sandbox_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sandbox: {str(e)}")

@router.post("/sandboxes/{sandbox_id}/files")
async def create_file(
    sandbox_id: str, 
    path: str = Form(...),
    file: UploadFile = File(...),
    request: Request = None,
    user_id: Optional[str] = Depends(get_optional_user_id)
):
    """Create a file in the sandbox using direct file upload"""
    logger.info(f"Received file upload request for sandbox {sandbox_id}, path: {path}, user_id: {user_id}")
    client = await db.client
    
    # Verify the user has access to this sandbox
    await verify_sandbox_access(client, sandbox_id, user_id)
    
    try:
        # Get sandbox using the safer method
        sandbox = await get_sandbox_by_id_safely(client, sandbox_id)
        
        # Read file content directly from the uploaded file
        content = await file.read()
        
        # Create file using raw binary content
        sandbox.fs.upload_file(path, content)
        logger.info(f"File created at {path} in sandbox {sandbox_id}")
        
        return {"status": "success", "created": True, "path": path}
    except Exception as e:
        logger.error(f"Error creating file in sandbox {sandbox_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# For backward compatibility, keep the JSON version too
@router.post("/sandboxes/{sandbox_id}/files/json")
async def create_file_json(
    sandbox_id: str, 
    file_request: dict,
    request: Request = None,
    user_id: Optional[str] = Depends(get_optional_user_id)
):
    """Create a file in the sandbox using JSON (legacy support)"""
    logger.info(f"Received JSON file creation request for sandbox {sandbox_id}, user_id: {user_id}")
    client = await db.client
    
    # Verify the user has access to this sandbox
    await verify_sandbox_access(client, sandbox_id, user_id)
    
    try:
        # Get sandbox using the safer method
        sandbox = await get_sandbox_by_id_safely(client, sandbox_id)
        
        # Get file path and content
        path = file_request.get("path")
        content = file_request.get("content", "")
        
        if not path:
            logger.error(f"Missing file path in request for sandbox {sandbox_id}")
            raise HTTPException(status_code=400, detail="File path is required")
        
        # Convert string content to bytes
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        # Create file
        sandbox.fs.upload_file(path, content)
        logger.info(f"File created at {path} in sandbox {sandbox_id}")
        
        return {"status": "success", "created": True, "path": path}
    except Exception as e:
        logger.error(f"Error creating file in sandbox {sandbox_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sandboxes/{sandbox_id}/files")
async def list_files(
    sandbox_id: str, 
    path: str,
    request: Request = None,
    user_id: Optional[str] = Depends(get_optional_user_id)
):
    """List files and directories at the specified path"""
    logger.info(f"Received list files request for sandbox {sandbox_id}, path: {path}, user_id: {user_id}")
    client = await db.client
    
    # Verify the user has access to this sandbox
    await verify_sandbox_access(client, sandbox_id, user_id)
    
    try:
        # Get sandbox using the safer method
        sandbox = await get_sandbox_by_id_safely(client, sandbox_id)
        
        # List files
        files = sandbox.fs.list_files(path)
        result = []
        
        for file in files:
            # Convert file information to our model
            # Ensure forward slashes are used for paths, regardless of OS
            full_path = f"{path.rstrip('/')}/{file.name}" if path != '/' else f"/{file.name}"
            file_info = FileInfo(
                name=file.name,
                path=full_path, # Use the constructed path
                is_dir=file.is_dir,
                size=file.size,
                mod_time=str(file.mod_time),
                permissions=getattr(file, 'permissions', None)
            )
            result.append(file_info)
        
        logger.info(f"Successfully listed {len(result)} files in sandbox {sandbox_id}")
        return {"files": [file.dict() for file in result]}
    except Exception as e:
        logger.error(f"Error listing files in sandbox {sandbox_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sandboxes/{sandbox_id}/files/content")
async def read_file(
    sandbox_id: str, 
    path: str,
    request: Request = None,
    user_id: Optional[str] = Depends(get_optional_user_id)
):
    """Read a file from the sandbox"""
    logger.info(f"Received file read request for sandbox {sandbox_id}, path: {path}, user_id: {user_id}")
    client = await db.client
    
    # Verify the user has access to this sandbox
    await verify_sandbox_access(client, sandbox_id, user_id)
    
    try:
        # Get sandbox using the safer method
        sandbox = await get_sandbox_by_id_safely(client, sandbox_id)
        
        # Read file
        content = sandbox.fs.download_file(path)
        
        # Return a Response object with the content directly
        filename = os.path.basename(path)
        logger.info(f"Successfully read file {filename} from sandbox {sandbox_id}")
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Error reading file in sandbox {sandbox_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/project/{project_id}/sandbox/ensure-active")
async def ensure_project_sandbox_active(
    project_id: str,
    request: Request = None,
    user_id: Optional[str] = Depends(get_optional_user_id)
):
    """
    Ensure that a project's sandbox is active and running.
    Checks the sandbox status and starts it if it's not running.
    """
    logger.info(f"Received ensure sandbox active request for project {project_id}, user_id: {user_id}")
    client = await db.client
    
    # Find the project and sandbox information
    project_result = await client.table('projects').select('*').eq('project_id', project_id).execute()
    
    if not project_result.data or len(project_result.data) == 0:
        logger.error(f"Project not found: {project_id}")
        raise HTTPException(status_code=404, detail="Project not found")
    
    project_data = project_result.data[0]
    
    # For public projects, no authentication is needed
    if not project_data.get('is_public'):
        # For private projects, we must have a user_id
        if not user_id:
            logger.error(f"Authentication required for private project {project_id}")
            raise HTTPException(status_code=401, detail="Authentication required for this resource")
            
        account_id = project_data.get('account_id')
        
        # Verify account membership
        if account_id:
            account_user_result = await client.schema('basejump').from_('account_user').select('account_role').eq('user_id', user_id).eq('account_id', account_id).execute()
            if not (account_user_result.data and len(account_user_result.data) > 0):
                logger.error(f"User {user_id} not authorized to access project {project_id}")
                raise HTTPException(status_code=403, detail="Not authorized to access this project")
    
    try:
        # Get or create the sandbox
        logger.info(f"Ensuring sandbox is active for project {project_id}")
        sandbox, sandbox_id, sandbox_pass = await get_or_create_project_sandbox(client, project_id)
        
        logger.info(f"Successfully ensured sandbox {sandbox_id} is active for project {project_id}")
        
        return {
            "status": "success", 
            "sandbox_id": sandbox_id,
            "message": "Sandbox is active"
        }
    except Exception as e:
        logger.error(f"Error ensuring sandbox is active for project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _recursive_list_files(
    sandbox_fs: any, base_path: str, current_relative_path: str = ""
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    Recursively lists files in the sandbox, yielding tuples of (full_path_in_sandbox, arcname_for_zip).
    Ensures that paths are correctly formatted for both sandbox access and zip archive.
    Args:
        sandbox_fs: The sandbox filesystem object (e.g., sandbox.fs).
        base_path: The absolute base path in the sandbox to start listing from (e.g., '/workspace').
        current_relative_path: The current path relative to base_path, used for recursion and arcname.
    """
    # Normalize base_path to ensure it ends with a slash if it's not empty, for consistent joining
    if base_path and not base_path.endswith('/'):
        normalized_base_path = base_path + '/'
    else:
        normalized_base_path = base_path

    # Construct the full path to list in the sandbox
    # If current_relative_path is empty, list normalized_base_path itself (or just / if base_path is /)
    # Otherwise, join normalized_base_path with current_relative_path
    if not current_relative_path:
        path_to_list = normalized_base_path.rstrip('/') if normalized_base_path != '/' else '/'
    else:
        # Ensure no leading slash on current_relative_path for joining
        path_to_list = os.path.join(normalized_base_path.rstrip('/'), current_relative_path.lstrip('/'))
    
    # Ensure path_to_list is correctly formatted, especially for root.
    # os.path.join might remove trailing slash if base_path is root and current_relative_path is empty.
    if base_path == '/' and not current_relative_path:
        path_to_list = '/'
    elif not path_to_list.startswith('/') and path_to_list: # Ensure it's an absolute path if not empty
        path_to_list = '/' + path_to_list

    logger.debug(f"Recursively listing files in: {path_to_list}")

    try:
        items = sandbox_fs.list_files(path_to_list)
        for item in items:
            # Construct the new relative path for the item for the next recursion level or for zipping
            item_relative_path = os.path.join(current_relative_path, item.name)
            full_path_in_sandbox = os.path.join(path_to_list, item.name)
            # Ensure full_path_in_sandbox is correctly joined if path_to_list is root
            if path_to_list == '/':
                 full_path_in_sandbox = f"/{item.name}"

            if item.is_dir:
                async for sub_item_path, sub_arcname in _recursive_list_files(
                    sandbox_fs, base_path, item_relative_path
                ):
                    yield sub_item_path, sub_arcname
            else:
                # arcname should be relative to the initial base_path (e.g. /workspace)
                # item_relative_path is already correctly relative to base_path
                yield full_path_in_sandbox, item_relative_path
    except Exception as e:
        logger.error(f"Error listing files in '{path_to_list}': {e}")
        # Depending on desired behavior, you might want to raise or just log and continue

@router.get("/sandboxes/{sandbox_id}/workspace/zip")
async def zip_workspace_files(
    sandbox_id: str,
    path: Optional[str] = None, # Optional path to zip, defaults to /workspace
    request: Request = None,
    user_id: Optional[str] = Depends(get_optional_user_id)
):
    """Create and stream a ZIP archive of the workspace files."""
    logger.info(f"Received request to ZIP workspace for sandbox {sandbox_id}, path: {path or '/workspace'}, user_id: {user_id}")
    client = await db.client

    project_data = await verify_sandbox_access(client, sandbox_id, user_id)
    project_name = project_data.get('name', 'workspace')
    # Sanitize project_name for filename
    safe_project_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in project_name)
    zip_filename = f"{safe_project_name}_workspace.zip"

    try:
        sandbox = await get_sandbox_by_id_safely(client, sandbox_id)
    except HTTPException as e:
        # If sandbox retrieval itself fails, re-raise the HTTP exception
        raise e
    except Exception as e:
        logger.error(f"Unexpected error getting sandbox {sandbox_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not access sandbox: {str(e)}")

    zip_buffer = io.BytesIO()
    base_zip_path = (path or '/workspace').rstrip('/')
    if not base_zip_path.startswith('/'): # Ensure base_zip_path is absolute
        base_zip_path = '/' + base_zip_path
    if not base_zip_path: # Handle case if path was just '/'
        base_zip_path = '/'

    # Define an async generator for streaming the ZIP content
    async def file_stream_generator():
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            try:
                async for file_path_in_sandbox, arcname_in_zip in _recursive_list_files(sandbox.fs, base_zip_path):
                    logger.debug(f"Adding to ZIP: {file_path_in_sandbox} as {arcname_in_zip}")
                    try:
                        file_content = sandbox.fs.download_file(file_path_in_sandbox)
                        zf.writestr(arcname_in_zip, file_content)
                    except Exception as e:
                        logger.error(f"Could not read or add file {file_path_in_sandbox} to ZIP: {e}")
                        # Add an empty file or a file with error message instead? For now, skip.
            except Exception as e:
                logger.error(f"Error during ZIP creation for sandbox {sandbox_id}: {e}")
                # If an error occurs during generation, this will be caught by the main try/except
                # and a 500 error will be returned before headers are sent.

        # After zf is closed (all files written to zip_buffer)
        zip_buffer.seek(0)
        # Yield chunks of the zip_buffer
        chunk_size = 8192
        while True:
            chunk = zip_buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk
        zip_buffer.close() # Ensure buffer is closed

    try:
        return StreamingResponse(
            file_stream_generator(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
    except Exception as e:
        logger.error(f"Failed to stream ZIP for sandbox {sandbox_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate or stream ZIP archive.")
