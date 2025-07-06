import pytest
import wikipedia
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, '/Users/Nobel.Khandaker/sources/mcp-assistant/src')

from mcp_server import (
    highlight_sections_prompt,
    fetch_wikipedia_info,
    list_wikipedia_sections,
    get_section_content
)


class TestHighlightSectionsPrompt:
    """Test the highlight_sections_prompt function."""
    
    def test_highlight_sections_prompt_basic(self):
        """Test basic prompt generation."""
        topic = "Python Programming"
        result = highlight_sections_prompt(topic)
        
        assert isinstance(result, str)
        assert "Python Programming" in result
        assert "section titles" in result
        assert "3–5 most important" in result
    
    def test_highlight_sections_prompt_special_characters(self):
        """Test prompt with special characters in topic."""
        topic = "C++ & Object-Oriented Programming"
        result = highlight_sections_prompt(topic)
        
        assert isinstance(result, str)
        assert "C++ & Object-Oriented Programming" in result


class TestFetchWikipediaInfo:
    """Test the fetch_wikipedia_info function."""
    
    @patch('mcp_server.wikipedia.search')
    @patch('mcp_server.wikipedia.page')
    def test_fetch_wikipedia_info_success(self, mock_page, mock_search):
        """Test successful Wikipedia info fetching."""
        mock_search.return_value = ["Python (programming language)"]
        
        mock_page_obj = Mock()
        mock_page_obj.title = "Python (programming language)"
        mock_page_obj.summary = "Python is a high-level programming language."
        mock_page_obj.url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        mock_page.return_value = mock_page_obj
        
        result = fetch_wikipedia_info("Python programming")
        
        assert "error" not in result
        assert result["title"] == "Python (programming language)"
        assert result["summary"] == "Python is a high-level programming language."
        assert result["url"] == "https://en.wikipedia.org/wiki/Python_(programming_language)"
        
        mock_search.assert_called_once_with("Python programming")
        mock_page.assert_called_once_with("Python (programming language)")
    
    @patch('mcp_server.wikipedia.search')
    def test_fetch_wikipedia_info_no_results(self, mock_search):
        """Test handling of no search results."""
        mock_search.return_value = []
        
        result = fetch_wikipedia_info("nonexistent topic")
        
        assert "error" in result
        assert result["error"] == "No results found for your query."
    
    @patch('mcp_server.wikipedia.search')
    @patch('mcp_server.wikipedia.page')
    def test_fetch_wikipedia_info_disambiguation_error(self, mock_page, mock_search):
        """Test handling of disambiguation error."""
        mock_search.return_value = ["Python"]
        
        disambiguation_error = wikipedia.DisambiguationError(
            "Python", 
            ["Python (programming language)", "Python (mythology)", "Python (snake)"]
        )
        mock_page.side_effect = disambiguation_error
        
        result = fetch_wikipedia_info("Python")
        
        assert "error" in result
        assert "Ambiguous topic" in result["error"]
        assert "Python (programming language)" in result["error"]
    
    @patch('mcp_server.wikipedia.search')
    @patch('mcp_server.wikipedia.page')
    def test_fetch_wikipedia_info_page_error(self, mock_page, mock_search):
        """Test handling of page error."""
        mock_search.return_value = ["InvalidPage"]
        mock_page.side_effect = wikipedia.PageError("InvalidPage")
        
        result = fetch_wikipedia_info("InvalidPage")
        
        assert "error" in result
        assert result["error"] == "No Wikipedia page could be loaded for this query."


class TestListWikipediaSections:
    """Test the list_wikipedia_sections function."""
    
    @patch('mcp_server.wikipedia.page')
    def test_list_wikipedia_sections_success(self, mock_page):
        """Test successful section listing."""
        mock_page_obj = Mock()
        mock_page_obj.sections = ["Introduction", "History", "Features", "Syntax"]
        mock_page.return_value = mock_page_obj
        
        result = list_wikipedia_sections("Python programming")
        
        assert "error" not in result
        assert "sections" in result
        assert result["sections"] == ["Introduction", "History", "Features", "Syntax"]
        mock_page.assert_called_once_with("Python programming")
    
    @patch('mcp_server.wikipedia.page')
    def test_list_wikipedia_sections_disambiguation_error(self, mock_page):
        """Test handling of disambiguation error."""
        disambiguation_error = wikipedia.DisambiguationError(
            "Python", 
            ["Python (programming language)", "Python (mythology)"]
        )
        mock_page.side_effect = disambiguation_error
        
        result = list_wikipedia_sections("Python")
        
        assert "error" in result
        assert "Ambiguous topic" in result["error"]
        assert "Python (programming language)" in result["error"]
    
    @patch('mcp_server.wikipedia.page')
    def test_list_wikipedia_sections_page_error(self, mock_page):
        """Test handling of page error."""
        mock_page.side_effect = wikipedia.PageError("InvalidPage")
        
        result = list_wikipedia_sections("InvalidPage")
        
        assert "error" in result
        assert result["error"] == "No Wikipedia page could be loaded for this query."
    
    @patch('mcp_server.wikipedia.page')
    def test_list_wikipedia_sections_generic_error(self, mock_page):
        """Test handling of generic error."""
        mock_page.side_effect = Exception("Network error")
        
        result = list_wikipedia_sections("Python")
        
        assert "error" in result
        assert result["error"] == "Network error"


class TestGetSectionContent:
    """Test the get_section_content function."""
    
    @patch('mcp_server.wikipedia.page')
    def test_get_section_content_success(self, mock_page):
        """Test successful section content retrieval."""
        mock_page_obj = Mock()
        mock_page_obj.section.return_value = "This is the history section content."
        mock_page.return_value = mock_page_obj
        
        result = get_section_content("Python programming", "History")
        
        assert "error" not in result
        assert "content" in result
        assert result["content"] == "This is the history section content."
        mock_page.assert_called_once_with("Python programming")
        mock_page_obj.section.assert_called_once_with("History")
    
    @patch('mcp_server.wikipedia.page')
    def test_get_section_content_section_not_found(self, mock_page):
        """Test handling of section not found."""
        mock_page_obj = Mock()
        mock_page_obj.section.return_value = None
        mock_page.return_value = mock_page_obj
        
        result = get_section_content("Python programming", "NonexistentSection")
        
        assert "error" in result
        assert "Section 'NonexistentSection' not found" in result["error"]
        assert "Python programming" in result["error"]
    
    @patch('mcp_server.wikipedia.page')
    def test_get_section_content_disambiguation_error(self, mock_page):
        """Test handling of disambiguation error."""
        disambiguation_error = wikipedia.DisambiguationError(
            "Python", 
            ["Python (programming language)", "Python (mythology)"]
        )
        mock_page.side_effect = disambiguation_error
        
        result = get_section_content("Python", "History")
        
        assert "error" in result
        assert "Ambiguous topic" in result["error"]
        assert "Python (programming language)" in result["error"]
    
    @patch('mcp_server.wikipedia.page')
    def test_get_section_content_page_error(self, mock_page):
        """Test handling of page error."""
        mock_page.side_effect = wikipedia.PageError("InvalidPage")
        
        result = get_section_content("InvalidPage", "History")
        
        assert "error" in result
        assert result["error"] == "No Wikipedia page could be loaded for this query."
    
    @patch('mcp_server.wikipedia.page')
    def test_get_section_content_generic_error(self, mock_page):
        """Test handling of generic error."""
        mock_page.side_effect = Exception("Network error")
        
        result = get_section_content("Python", "History")
        
        assert "error" in result
        assert result["error"] == "Network error"


if __name__ == "__main__":
    pytest.main([__file__])