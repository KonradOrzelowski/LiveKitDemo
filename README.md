# LiveKitDemo
This repository demonstrates the usage of an agent capable of responding to questions based on a lesson conspect. The agent can listen to user speech as well as read what the user writes. It responds to the user by speaking and also outputs text.

Answering questions is done by adding a tool that calls another model to find the most relevant documents and then generates a response based on them. This approach makes the entire RAG system more modular. For example, one team can focus on question answering while another works on the main components of the model. Similar approach was used for other parts of the project as OpenAI handle general tasks, Deepgram uses AI to make transcription of text, while Cartesia is used to generate spoken messages.

Additionally, adding tools like this is an easy way to incorporate other functionalities, such as visual recognition, by integrating other models as tools. For example, LangChain by default can't process visual processing, and it's necessary to use OpenAI library.

By assigning separate models for each task, it is possible to use a simpler model for answering questions based on documents, while reserving more advanced models for general-purpose tasks. This approach helps reduce the cost of handling simpler queries.

For example:
- small model like gpt-3.5 or gpt-4o-mini can be used for simple doc questions
- bigger model like gpt-4o can be used when need smarter thinking or work with images or audio


This setup help with:
- saving money on easy questions
- using strong model only when really needed
- changing or upgrading one part without breaking rest






