import chainlit as cl
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig


gemini_api_key = os.getenv("GEMINI_API_KEY")


# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant")

# result = Runner.run_sync(agent, "Tell me a joke", run_config=config)

# print(result.final_output)


@cl.on_chat_start
async def handle_chat_start():

    cl.user_session.set("history", [])
    await cl.Message(content="Hello! How can I help you today?").send()    

@cl.on_message
async def handle_message(message: cl.Message):

    history = cl.user_session.get("history") 
    history.append({"role": "user", "content": message.content})
   

    result = await Runner.run(
        agent,
        input= history,
        run_config= config
    )

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()

