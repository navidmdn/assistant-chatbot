import wikipediaapi

USER_AGENT_HEADER = 'RAG-wiki-chatbot (smadani@buffalo.edu)'

wiki_wiki = wikipediaapi.Wikipedia(USER_AGENT_HEADER, 'en')
page_py = wiki_wiki.page('Barack Obama')
print(page_py.summary)