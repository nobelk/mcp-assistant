import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import List

import sys
sys.path.insert(0, '/Users/Nobel.Khandaker/sources/mcp-assistant/src')

from mcp_client import load_mcp_tools, create_graph, list_prompts, handle_prompt


class TestLoadMCPTools:
    """Test the load_mcp_tools function."""
    
    @pytest.mark.asyncio
    async def test_load_mcp_tools_success(self):
        """Test successful loading of MCP tools."""
        mock_session = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        
        mock_tools_response = Mock()
        mock_tools_response.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_tools_response
        
        mock_result = Mock()
        mock_result.content = [Mock(text="test result")]
        mock_session.call_tool.return_value = mock_result
        
        tools = await load_mcp_tools(mock_session)
        
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert tools[0].description == "Test tool description"
        mock_session.list_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_mcp_tools_empty_response(self):
        """Test handling of empty tools response."""
        mock_session = AsyncMock()
        mock_tools_response = Mock()
        mock_tools_response.tools = []
        mock_session.list_tools.return_value = mock_tools_response
        
        tools = await load_mcp_tools(mock_session)
        
        assert len(tools) == 0
        mock_session.list_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_mcp_tools_tool_execution_error(self):
        """Test tool execution with error handling."""
        mock_session = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "error_tool"
        mock_tool.description = "Error tool"
        
        mock_tools_response = Mock()
        mock_tools_response.tools = [mock_tool]
        mock_session.list_tools.return_value = mock_tools_response
        
        mock_session.call_tool.side_effect = Exception("Tool execution failed")
        
        tools = await load_mcp_tools(mock_session)
        tool_func = tools[0].func
        
        result = await tool_func("test input")
        assert "Error calling tool error_tool: Tool execution failed" in result


class TestCreateGraph:
    """Test the create_graph function."""
    
    @pytest.mark.asyncio
    @patch('mcp_client.load_mcp_tools')
    @patch('mcp_client.ChatOpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    async def test_create_graph_success(self, mock_chat_openai, mock_load_tools):
        """Test successful graph creation."""
        mock_session = AsyncMock()
        mock_tools = []  # Empty tools list to avoid ToolNode issues
        mock_load_tools.return_value = mock_tools
        
        mock_llm = Mock()
        mock_llm.bind_tools.return_value = Mock()
        mock_chat_openai.return_value = mock_llm
        
        graph = await create_graph(mock_session)
        
        assert graph is not None
        mock_load_tools.assert_called_once_with(mock_session)
        mock_chat_openai.assert_called_once_with(
            model="gpt-4", 
            temperature=0, 
            openai_api_key="test_key"
        )
    
    @pytest.mark.asyncio
    @patch('mcp_client.load_mcp_tools')
    @patch('mcp_client.ChatOpenAI')
    @patch.dict(os.environ, {}, clear=True)
    async def test_create_graph_missing_api_key(self, mock_chat_openai, mock_load_tools):
        """Test graph creation with missing API key."""
        mock_session = AsyncMock()
        mock_tools = []
        mock_load_tools.return_value = mock_tools
        
        mock_llm = Mock()
        mock_llm.bind_tools.return_value = Mock()
        mock_chat_openai.return_value = mock_llm
        
        graph = await create_graph(mock_session)
        
        mock_chat_openai.assert_called_once_with(
            model="gpt-4", 
            temperature=0, 
            openai_api_key=None
        )


class TestListPrompts:
    """Test the list_prompts function."""
    
    @pytest.mark.asyncio
    async def test_list_prompts_success(self, capsys):
        """Test successful listing of prompts."""
        mock_session = AsyncMock()
        mock_prompt = Mock()
        mock_prompt.name = "test_prompt"
        
        mock_arg1 = Mock()
        mock_arg1.name = "arg1"
        mock_arg2 = Mock()
        mock_arg2.name = "arg2"
        mock_prompt.arguments = [mock_arg1, mock_arg2]
        
        mock_response = Mock()
        mock_response.prompts = [mock_prompt]
        mock_session.list_prompts.return_value = mock_response
        
        await list_prompts(mock_session)
        
        captured = capsys.readouterr()
        assert "Available Prompts and Argument Structure:" in captured.out
        assert "Prompt: test_prompt" in captured.out
        assert "- arg1" in captured.out
        assert "- arg2" in captured.out
    
    @pytest.mark.asyncio
    async def test_list_prompts_empty(self, capsys):
        """Test handling of empty prompts response."""
        mock_session = AsyncMock()
        mock_response = Mock()
        mock_response.prompts = []
        mock_session.list_prompts.return_value = mock_response
        
        await list_prompts(mock_session)
        
        captured = capsys.readouterr()
        assert "No prompts found on the server." in captured.out
    
    @pytest.mark.asyncio
    async def test_list_prompts_no_arguments(self, capsys):
        """Test prompt with no arguments."""
        mock_session = AsyncMock()
        mock_prompt = Mock()
        mock_prompt.name = "simple_prompt"
        mock_prompt.arguments = []
        
        mock_response = Mock()
        mock_response.prompts = [mock_prompt]
        mock_session.list_prompts.return_value = mock_response
        
        await list_prompts(mock_session)
        
        captured = capsys.readouterr()
        assert "Prompt: simple_prompt" in captured.out
        assert "- No arguments required." in captured.out


class TestHandlePrompt:
    """Test the handle_prompt function."""
    
    @pytest.mark.asyncio
    async def test_handle_prompt_invalid_usage(self, capsys):
        """Test handling of invalid prompt usage."""
        mock_session = AsyncMock()
        mock_tools = []
        mock_agent = AsyncMock()
        
        await handle_prompt(mock_session, mock_tools, "/prompt", mock_agent)
        
        captured = capsys.readouterr()
        assert "Usage: /prompt <name>" in captured.out
    
    @pytest.mark.asyncio
    async def test_handle_prompt_not_found(self, capsys):
        """Test handling of non-existent prompt."""
        mock_session = AsyncMock()
        mock_tools = []
        mock_agent = AsyncMock()
        
        mock_response = Mock()
        mock_response.prompts = []
        mock_session.list_prompts.return_value = mock_response
        
        await handle_prompt(mock_session, mock_tools, "/prompt nonexistent", mock_agent)
        
        captured = capsys.readouterr()
        assert "Prompt 'nonexistent' not found." in captured.out
    
    @pytest.mark.asyncio
    async def test_handle_prompt_wrong_arg_count(self, capsys):
        """Test handling of wrong argument count."""
        mock_session = AsyncMock()
        mock_tools = []
        mock_agent = AsyncMock()
        
        mock_prompt = Mock()
        mock_prompt.name = "test_prompt"
        
        mock_arg1 = Mock()
        mock_arg1.name = "arg1"
        mock_arg2 = Mock()
        mock_arg2.name = "arg2"
        mock_prompt.arguments = [mock_arg1, mock_arg2]
        
        mock_response = Mock()
        mock_response.prompts = [mock_prompt]
        mock_session.list_prompts.return_value = mock_response
        
        await handle_prompt(mock_session, mock_tools, "/prompt test_prompt 'only_one_arg'", mock_agent)
        
        captured = capsys.readouterr()
        assert "Expected 2 arguments" in captured.out
    
    @pytest.mark.asyncio
    async def test_handle_prompt_execution_error(self, capsys):
        """Test handling of prompt execution error."""
        mock_session = AsyncMock()
        mock_tools = []
        mock_agent = AsyncMock()
        
        mock_prompt = Mock()
        mock_prompt.name = "test_prompt"
        mock_prompt.arguments = []
        
        mock_response = Mock()
        mock_response.prompts = [mock_prompt]
        mock_session.list_prompts.return_value = mock_response
        
        mock_session.get_prompt.side_effect = Exception("Execution failed")
        
        await handle_prompt(mock_session, mock_tools, "/prompt test_prompt", mock_agent)
        
        captured = capsys.readouterr()
        assert "Prompt invocation failed: Execution failed" in captured.out


if __name__ == "__main__":
    pytest.main([__file__])