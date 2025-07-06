import wikipedia
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WikipediaSearch")

@mcp.prompt()
def highlight_sections_prompt(topic: str) -> str:
    """
    Identifies the most important sections from a Wikipedia article on the given topic.
    """
    return f"""
    The user is exploring the Wikipedia article on "{topic}".

    Given the list of section titles from the article, choose the 3–5 most important or interesting sections 
    that are likely to help someone learn about the topic.

    Return a bullet list of these section titles, along with 1-line explanations of why each one matters.
    """


@mcp.tool()
def fetch_wikipedia_info(query: str) -> dict:
    """
    Search Wikipedia for a topic and return title, summary, and URL of the best match.
    """
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return {"error": "No results found for your query."}

        best_match = search_results[0]
        page = wikipedia.page(best_match)

        return {
            "title": page.title,
            "summary": page.summary,
            "url": page.url
        }

    except wikipedia.DisambiguationError as e:
        return {
            "error": f"Ambiguous topic. Try one of these: {', '.join(e.options[:5])}"
        }

    except wikipedia.PageError:
        return {
            "error": "No Wikipedia page could be loaded for this query."
        }

@mcp.tool()
def list_wikipedia_sections(topic: str) -> dict:
    """
    Return a list of section titles from the Wikipedia page of a given topic.
    """
    try:
        page = wikipedia.page(topic)
        sections = page.sections
        return {"sections": sections}
    except wikipedia.DisambiguationError as e:
        return {
            "error": f"Ambiguous topic. Try one of these: {', '.join(e.options[:5])}"
        }
    except wikipedia.PageError:
        return {
            "error": "No Wikipedia page could be loaded for this query."
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_section_content(topic: str, section_title: str) -> dict:
    """
    Return the content of a specific section in a Wikipedia article.
    """
    try:
        page = wikipedia.page(topic)
        content = page.section(section_title)
        if content:
            return {"content": content}
        else:
            return {"error": f"Section '{section_title}' not found in article '{topic}'."}
    except wikipedia.DisambiguationError as e:
        return {
            "error": f"Ambiguous topic. Try one of these: {', '.join(e.options[:5])}"
        }
    except wikipedia.PageError:
        return {
            "error": "No Wikipedia page could be loaded for this query."
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting MCP Wikipedia Server with multiple tools...")
    mcp.run(transport="stdio")