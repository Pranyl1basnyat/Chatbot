import os
from typing import Any, List, Optional
from pydantic import Field
from dotenv import load_dotenv, find_dotenv

# LangChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM

# Hugging Face
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings


# =========================
# ✅ LOAD ENV
# =========================
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")


# =========================
# ✅ LLM CREATOR
# =========================
def create_llm():
    print("\n🚀 Attempting to create LLM using multiple strategies...")

    # ====================
    # STRATEGY 1: Local Transformers
    # ====================
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch

        print("\n🔄 Strategy 1: Local Transformers")
        print(" - Loading DistilGPT2 locally...")

        try:
            pipe = pipeline(
                'text-generation',
                model='distilgpt2',
                tokenizer='distilgpt2',
                device=-1,
                torch_dtype=torch.float32,
                pad_token_id=50256
            )

            class LocalPipelineLLM(LLM):
                pipeline: Any = Field()

                def __init__(self, pipeline, **data):
                    super().__init__(**data)
                    object.__setattr__(self, "pipeline", pipeline)

                @property
                def _llm_type(self) -> str:
                    return "local_pipeline"

                def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> str:
                    prompt = prompt.strip()
                    if not prompt:
                        return "Please enter a question."

                    result = self.pipeline(
                        prompt,
                        max_length=min(len(prompt.split()) + 50, 150),
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=50256,
                        eos_token_id=50256,
                        repetition_penalty=1.1
                    )

                    generated = result[0]['generated_text']
                    response = generated[len(prompt):].strip()

                    if stop:
                        for stop_seq in stop:
                            if stop_seq in response:
                                response = response.split(stop_seq)[0]

                    return response or "Sorry, I couldn't generate a response."

            print("✅ Local pipeline LLM created")
            return LocalPipelineLLM(pipe)

        except Exception as e:
            print(f"❌ Pipeline approach failed: {e}")

        print("\n - Trying direct model loading...")
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        model = AutoModelForCausalLM.from_pretrained('distilgpt2')

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        class LocalDirectLLM(LLM):
            model: Any = Field()
            tokenizer: Any = Field()

            def __init__(self, model, tokenizer, **data):
                super().__init__(**data)
                object.__setattr__(self, "model", model)
                object.__setattr__(self, "tokenizer", tokenizer)
                self.model.eval()

            @property
            def _llm_type(self) -> str:
                return "local_direct"

            def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> str:
                prompt = prompt.strip()
                inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=100, truncation=True)

                import torch
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 30,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()

                if stop:
                    for stop_seq in stop:
                        if stop_seq in response:
                            response = response.split(stop_seq)[0]

                return response or "Sorry, I couldn't generate a response."

        print("✅ Local direct LLM created")
        return LocalDirectLLM(model, tokenizer)

    except ImportError:
        print("❌ Transformers not installed. Install with: pip install transformers torch")
    except Exception as e:
        print(f"❌ Local transformers failed: {e}")

    # ====================
    # STRATEGY 2: HuggingFace Endpoint
    # ====================
    print("\n🔄 Strategy 2: HuggingFace Endpoints")
    if HF_TOKEN:
        print(f" - Token detected: hf_***{HF_TOKEN[-5:]}")
        try:
            import requests
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            response = requests.get("https://huggingface.co/api/whoami", headers=headers, timeout=10)
            if response.status_code == 200:
                print("✅ Token is valid")

                models_to_try = [
                    "microsoft/DialoGPT-small",
                    "distilgpt2",
                    "gpt2",
                ]

                for model_name in models_to_try:
                    try:
                        print(f" - Trying HF endpoint: {model_name}")
                        llm = HuggingFaceEndpoint(
                            repo_id=model_name,
                            temperature=0.7,
                            max_new_tokens=100,
                            huggingfacehub_api_token=HF_TOKEN,
                            timeout=20
                        )
                        llm.invoke("Hi")
                        print(f"✅ HF endpoint works: {model_name}")
                        return llm

                    except Exception as e:
                        print(f"❌ {model_name} failed: {str(e)[:100]}...")
                        continue
            else:
                print(f"❌ Token invalid: {response.status_code}")
        except Exception as e:
            print(f"❌ Token check failed: {e}")
    else:
        print("❌ No Hugging Face token found in environment")

    # ====================
    # STRATEGY 3: Simple Fallback
    # ====================
    print("\n🔄 Strategy 3: Simple Fallback LLM")

    class SimpleFallbackLLM(LLM):
        responses: dict = Field(default_factory=lambda: {
            'hello': "Hello! I'm here to help answer your questions.",
            'hi': "Hi there! What would you like to know?",
            'what': "I can help answer questions about the provided context.",
            'how': "I can explain things based on the context.",
            'why': "I can provide explanations from the knowledge base.",
            'who': "I can identify people or entities in the context.",
            'when': "I can provide timing information.",
            'where': "I can provide location information."
        })

        @property
        def _llm_type(self) -> str:
            return "simple_fallback"

        def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> str:
            prompt = prompt.lower().strip()
            for keyword, response in self.responses.items():
                if keyword in prompt:
                    return response
            return "I understand your question. Based on the provided context, I'll try to help."

    print("✅ Simple fallback LLM created")
    return SimpleFallbackLLM()


# =========================
# ✅ CUSTOM PROMPT
# =========================
CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't know. Don't make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )


# =========================
# ✅ MAIN APP
# =========================
def main():
    print("\n✅ SCRIPT STARTED ✅\n")

    DB_FAISS_PATH = "vectorstore/db_faiss"

    print("🔹 Loading embedding model...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        print(" - Trying alternative model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    print("🔹 Checking FAISS database...")
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"❌ FAISS database not found at {DB_FAISS_PATH}")

    print("🔹 Loading FAISS vectorstore...")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    print("🔹 Creating LLM...")
    llm = create_llm()

    print("🔹 Building Retrieval QA Chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

    print("\n🎉 Chatbot is ready!")
    print("💡 Tip: Ask questions related to your knowledge base.")

    while True:
        user_query = input("\n🤖 Write Query Here (or 'quit' to exit): ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        if not user_query:
            print("⚠️ Please enter a valid query.")
            continue

        try:
            print("🔍 Processing your query...")
            response = qa_chain.invoke({'query': user_query})
            print(f"\n📋 RESULT:\n{response['result']}\n")
            print(f"\n📚 SOURCE DOCUMENTS ({len(response['source_documents'])}):")
            for i, doc in enumerate(response['source_documents'], 1):
                content = doc.page_content[:200].replace('\n', ' ')
                print(f"{i}. {content}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"   📄 Source: {doc.metadata}")
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("💡 Try asking a different question or check your database.")


# =========================
# ✅ RUN
# =========================
if __name__ == "__main__":
    main()
