import uuid
import sys
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from app.agent import app as adk_app

def main():
    session_service = InMemorySessionService()
    session_id = str(uuid.uuid4())
    
    runner = Runner(
        app=adk_app,
        session_service=session_service,
        auto_create_session=True,
    )
    
    new_message = types.Content(role="user", parts=[types.Part.from_text(text="https://www.kaggle.com/competitions/titanic")])
    
    print("Running...")
    try:
        for event in runner.run(
            user_id="test_user",
            session_id=session_id,
            new_message=new_message
        ):
            print(f"Event type: {type(event).__name__}")
            if hasattr(event, 'content'):
                print(f"Content: {event.content}")
            if hasattr(event, 'message'):
                print(f"Message: {event.message}")
            if hasattr(event, 'node_output'):
                print(f"Node Output: {event.node_output}")
            print(vars(event))
            print("---")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
