import os
import uuid
import streamlit as st

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from app.agent import app as adk_app

import json

CONVERSATIONS_FILE = "conversations.json"
WELCOME_MESSAGE = {"role": "assistant", "content": "👋 Welcome to the Kaggle Copilot! Please provide a valid Kaggle competition URL to get started."}

def load_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_conversation(session_id, messages):
    if not messages:
        return
    # Only save if there is at least one user message
    has_user_message = any(msg.get("role") == "user" for msg in messages)
    if not has_user_message:
        return
        
    convos = load_conversations()
    title = "New Conversation"
    for msg in messages:
        if msg["role"] == "user":
            title = msg["content"][:30] + ("..." if len(msg["content"]) > 30 else "")
            break
    convos[session_id] = {"title": title, "messages": messages}
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump(convos, f)

def delete_conversation(session_id):
    convos = load_conversations()
    if session_id in convos:
        del convos[session_id]
        with open(CONVERSATIONS_FILE, "w") as f:
            json.dump(convos, f)

st.set_page_config(page_title="Kaggle Copilot", layout="wide")

if "session_service" not in st.session_state:
    st.session_state.session_service = InMemorySessionService()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [WELCOME_MESSAGE]

# To resume the workflow on RequestInput
if "pending_interrupt_id" not in st.session_state:
    st.session_state.pending_interrupt_id = None

st.title("Kaggle Copilot Chat")

with st.sidebar:
    if st.button("Start New Conversation", type="primary", use_container_width=True):
        if "messages" in st.session_state and st.session_state.messages:
            save_conversation(st.session_state.session_id, st.session_state.messages)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [WELCOME_MESSAGE]
        st.session_state.pending_interrupt_id = None
        st.rerun()
        
    st.divider()
    st.subheader("Past Conversations")
    convos = load_conversations()
    if convos:
        for sid, data in reversed(list(convos.items())):
            if st.session_state.session_id == sid:
                # Active conversation: highlight and show delete button
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.button(f"**{data['title']}**", key=f"btn_{sid}_active", use_container_width=True, type="secondary")
                with col2:
                    if st.button("🗑️", key=f"del_{sid}", use_container_width=True, type="secondary"):
                        delete_conversation(sid)
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages = [WELCOME_MESSAGE]
                        st.session_state.pending_interrupt_id = None
                        st.rerun()
            else:
                # Inactive conversation: simple borderless text button
                if st.button(data["title"], key=f"btn_{sid}", use_container_width=True, type="tertiary"):
                    if "messages" in st.session_state and st.session_state.messages:
                        save_conversation(st.session_state.session_id, st.session_state.messages)
                    st.session_state.session_id = sid
                    st.session_state.messages = data["messages"]
                    st.session_state.pending_interrupt_id = None
                    st.rerun()
    else:
        st.info("No past conversations yet.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

user_input = st.chat_input("Message the copilot...", disabled=st.session_state.is_processing)

if user_input:
    st.session_state.pending_input = user_input
    st.session_state.is_processing = True
    st.rerun()

if st.session_state.is_processing and "pending_input" in st.session_state:
    input_text = st.session_state.pending_input
    del st.session_state["pending_input"]

    # 1. Store user message in history
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.markdown(input_text)

    # 2. Run the agent
    with st.chat_message("assistant"):
        st_placeholder = st.empty()
        full_response = ""

        runner = Runner(
            app=adk_app,
            session_service=st.session_state.session_service,
            auto_create_session=True,
        )

        new_message = types.Content(role="user", parts=[types.Part.from_text(text=input_text)])

        st.session_state.pending_interrupt_id = None

        try:
            for event in runner.run(
                user_id="streamlit_user",
                session_id=st.session_state.session_id,
                new_message=new_message
            ):
                is_request_input = False
                req_msg = "Need input"
                interrupt_id = None
                
                if type(event).__name__ == "RequestInput":
                    is_request_input = True
                    req_msg = getattr(event, "message", req_msg)
                    interrupt_id = getattr(event, "interrupt_id", None)
                
                content = getattr(event, "message", getattr(event, "content", None))
                if content and hasattr(content, "parts"):
                    for part in content.parts:
                        if hasattr(part, "text") and part.text:
                            text_val = part.text
                            if text_val.strip().startswith("{") and '"input_text"' in text_val:
                                author = getattr(event, "author", "Agent")
                                text_val = f"*{author} finished reasoning step.*\n"
                            
                            full_response += text_val + "\n"
                            st_placeholder.markdown(full_response + "▌")
                        if hasattr(part, "function_call") and part.function_call and part.function_call.name == "adk_request_input":
                            is_request_input = True
                            req_msg = part.function_call.args.get("message", req_msg)
                            interrupt_id = part.function_call.args.get("interruptId", None)
                
                if hasattr(event, "output") and event.output:
                    if isinstance(event.output, str):
                        full_response += "\n\n" + event.output + "\n"
                        st_placeholder.markdown(full_response + "▌")

                if is_request_input:
                    if interrupt_id:
                        st.session_state.pending_interrupt_id = interrupt_id
                    full_response += f"\n\n**{req_msg}**\n"
                    st_placeholder.markdown(full_response + "▌")
        except Exception as e:
            full_response += f"\n\nError: {e}"
        
        st_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_conversation(st.session_state.session_id, st.session_state.messages)
        
    st.session_state.is_processing = False
    st.rerun()
