This implementation is from llamaindex https://github.com/run-llama/openai_realtime_client.

The only change made is to remove the llamaindex wrapper abstraction and use the tool directly without llamaindex. The change is just in the tool definition. This involves writing the tool definition as the documentation of openai states, instead of using llamaindex (`tools = [FunctionTool.from_defaults(fn=get_phone_number)]`) : 

```python
tool = rag_tool = {
        "type": "function",
        "function": {
            "name": "RAG_tool",
            "description": "Get information about toulouse",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                Text query to search in the RAG database.
                                """,
                    }
                },
                "required": ["query"],
            },
        }
    }
```


Run the app with : 

```bash
python streaming_cli.py
```

I still need to add async call to the tool for now.