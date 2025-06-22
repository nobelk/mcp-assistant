import wikipedia

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WikipediaSearch")


@mcp.tool()
def fetch_wikipedia_info(query: str) -> dict:
    """
    Searches Wikipedia
    """
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return {"error": "No results found"}
        best_match = search_results[0]
        page = wikipedia.page(best_match)

        return {"title": page.title, "summary": page.summary, "url": page.url}
    except wikipedia.DisambiguationError as err:
        return {
            "error": f"Ambiguous topic. Try: {','.join(err.options[:3])}"
        }
    except wikipedia.PageError:
        return {
            "error": "No Wikipedia page found"
        }


@mcp.tool()
def list_wikipedia_sections(topic: str) -> dict:
    """
    Return a list of section titles
    """
    try:
        page = wikipedia.page(topic)
        sections = page.sections
        return {"sections": sections}
    except Exception as err:
        return {"error": str(err)}


@mcp.tool()
def get_section_content(topic: str, section_title: str) -> dict:
    """
    Return the content of a section
    """
    try:
        page = wikipedia.page
        content = page.section(section_title)
        if content:
            return {"content": content}
        else:
            return {"error": f"Section '{section_title}' not found in article"}
    except Exception as err:
        return {"error": str(err)}


if __name__ == "__main__":
    print("Starting MCP Wikipedia service ...")
    mcp.run(transport='stdio')
