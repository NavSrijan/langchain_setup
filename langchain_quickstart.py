"""
LangChain Financial Advisor Module
===================================
A standalone module for building financial AI agents using LangChain and SQLite.

Features:
- Custom tool creation and binding
- Multi-turn conversations with memory
- RAG (Retrieval Augmented Generation)
- Agent loops with tool calling
- Token logging and monitoring

Quick Import (Colab or local):
  from langchain_quickstart import Setup, Agent
  setup = Setup()
  agent = Agent(setup.get_user_id())
  response = agent.run_query("What's my spending?")
"""

import os
import json
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, Sequence, Annotated, Any, Dict
import operator
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
)
from langchain.memory import ConversationBufferWindowMemory
from langgraph.graph import StateGraph, END
from typing import TypedDict

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Numeric, 
    Enum, DateTime, ForeignKey, func, select, text
)
from sqlalchemy.orm import relationship, Session, declarative_base
import enum

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ============================================================================
# DATABASE MODELS AND SETUP
# ============================================================================

Base = declarative_base()

class TransactionType(enum.Enum):
    credit = "credit"
    debit = "debit"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    transactions = relationship("Transaction", back_populates="user")
    goals = relationship("Goal", back_populates="user")

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount = Column(Numeric, nullable=False)
    transaction_type = Column(Enum(TransactionType), nullable=False)
    transaction_date = Column(DateTime(timezone=True), nullable=True)
    description = Column(String, nullable=True)
    category = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="transactions")

class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    target_amount = Column(Numeric, nullable=False)
    current_amount = Column(Numeric, nullable=False, default=0)
    deadline = Column(DateTime(timezone=True), nullable=True)
    category = Column(String, nullable=True)
    priority = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="goals")

# ============================================================================
# DATABASE AND SETUP CLASSES
# ============================================================================

class DatabaseManager:
    """Handles SQLite database operations and session management."""
    
    def __init__(self, db_path: str = "financial_advisor.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        print(f"✓ Database initialized at {db_path}")
    
    def get_session(self) -> Session:
        from sqlalchemy.orm import sessionmaker
        SessionLocal = sessionmaker(bind=self.engine)
        return SessionLocal()
    
    def execute_query(self, query):
        session = self.get_session()
        try:
            result = session.execute(query)
            return result.all()
        finally:
            session.close()
    
    def add_user(self, username: str, email: str) -> User:
        session = self.get_session()
        try:
            user = User(username=username, email=email)
            session.add(user)
            session.commit()
            session.refresh(user)
            return user
        finally:
            session.close()
    
    def add_transaction(self, user_id: int, amount: float, transaction_type: str, 
                       category: str, description: str, days_ago: int = 0):
        session = self.get_session()
        try:
            tx_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
            transaction = Transaction(
                user_id=user_id,
                amount=Decimal(str(amount)),
                transaction_type=TransactionType[transaction_type],
                category=category,
                description=description,
                transaction_date=tx_date
            )
            session.add(transaction)
            session.commit()
            session.refresh(transaction)
            return transaction
        finally:
            session.close()
    
    def add_goal(self, user_id: int, title: str, target_amount: float, 
                 current_amount: float = 0, category: str = None, days_ahead: int = 365):
        session = self.get_session()
        try:
            deadline = datetime.now(timezone.utc) + timedelta(days=days_ahead)
            goal = Goal(
                user_id=user_id,
                title=title,
                target_amount=Decimal(str(target_amount)),
                current_amount=Decimal(str(current_amount)),
                category=category,
                deadline=deadline
            )
            session.add(goal)
            session.commit()
            session.refresh(goal)
            return goal
        finally:
            session.close()


class Setup:
    """Complete setup for a demo environment with pre-loaded data."""
    
    def __init__(self, db_path: str = "financial_advisor.db"):
        self.db = DatabaseManager(db_path)
        self.user = None
        self.initialize()
    
    def initialize(self):
        """Initialize database with sample user and data."""
        self.user = self.db.add_user("demo_user", "demo@example.com")
        print(f"✓ Demo user created (ID: {self.user.id})")
        self._create_sample_data()
    
    def _create_sample_data(self):
        """Populate database with sample transactions and goals."""
        categories = [
            ("food", "debit", 500),
            ("transport", "debit", 200),
            ("entertainment", "debit", 150),
            ("utilities", "debit", 300),
            ("salary", "credit", 50000),
        ]
        
        for category, tx_type, amount in categories:
            for day_offset in range(1, 31, 7):
                self.db.add_transaction(
                    user_id=self.user.id,
                    amount=amount,
                    transaction_type=tx_type,
                    category=category,
                    description=f"Sample {category} transaction",
                    days_ago=day_offset
                )
        
        self.db.add_goal(self.user.id, "Emergency Fund", 100000, 
                        current_amount=25000, category="savings")
        self.db.add_goal(self.user.id, "Vacation", 50000, 
                        current_amount=10000, category="travel", days_ahead=180)
        
        print(f"✓ Sample data created")
    
    def get_user_id(self) -> int:
        """Get the demo user ID for use in agents."""
        return self.user.id


# Global database manager
_db_manager: Optional[DatabaseManager] = None

def get_db() -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

# Custom Tools

class AggregateSpendByCategoryTool(BaseTool):
    name: str = "aggregate_spend_by_category"
    description: str = "Calculate total, average, or count of spends for a category. Input: {\"category\": \"food\", \"metric\": \"sum\", \"days\": 30}"
    
    user_id: int
    db: Any = None

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(kwargs))

    async def _arun(self, params: dict) -> str:
        try:
            category = params.get("category", "").strip()
            metric = params.get("metric", "sum")
            days = params.get("days")
            
            if not category or metric not in ["sum", "avg", "count"]:
                return "Error: category required, metric must be sum/avg/count"
            
            session = get_db().get_session()
            try:
                query = select(Transaction).where(
                    Transaction.user_id == self.user_id,
                    Transaction.category.ilike(f"%{category}%"),
                    Transaction.transaction_type == TransactionType.debit
                )
                
                if days:
                    since = datetime.now(timezone.utc) - timedelta(days=days)
                    query = query.where(Transaction.transaction_date >= since)
                
                transactions = session.execute(query).scalars().all()
                
                if not transactions:
                    return f"No {category} transactions found"
                
                amounts = [float(tx.amount) for tx in transactions]
                
                if metric == "sum":
                    result = sum(amounts)
                    unit = "Total"
                elif metric == "avg":
                    result = sum(amounts) / len(amounts)
                    unit = "Average"
                else:
                    result = len(amounts)
                    unit = "Count"
                
                period = f"last {days} days" if days else "all time"
                return f"{unit} {category} spend ({period}): ₹{result:.2f}"
            finally:
                session.close()
        except Exception as e:
            return f"Error: {str(e)}"


class ListCategoriesForUserTool(BaseTool):
    name: str = "list_categories"
    description: str = "Get all available transaction categories for the user. No input required."
    
    user_id: int

    def _run(self, **kwargs) -> str:
        session = get_db().get_session()
        try:
            query = select(Transaction.category).where(
                Transaction.user_id == self.user_id
            ).distinct()
            
            categories = session.execute(query).scalars().all()
            categories = [c for c in categories if c]
            
            if not categories:
                return "No categories found"
            return f"Available categories: {', '.join(categories)}"
        finally:
            session.close()

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class GoalProgressTool(BaseTool):
    name: str = "goal_progress"
    description: str = "Get progress on a financial goal. Input: {\"goal_id\": 1}"
    
    user_id: int

    def _run(self, goal_id: int) -> str:
        session = get_db().get_session()
        try:
            goal = session.execute(
                select(Goal).where(Goal.id == goal_id, Goal.user_id == self.user_id)
            ).scalar_one_or_none()
            
            if not goal:
                return "Goal not found"
            
            current = float(goal.current_amount)
            target = float(goal.target_amount)
            progress_pct = (current / target * 100) if target > 0 else 0
            remaining = target - current
            
            days_left = 0
            if goal.deadline:
                delta = goal.deadline - datetime.now(timezone.utc)
                days_left = delta.days
            
            return (
                f"Goal: {goal.title}\n"
                f"Progress: ₹{current:.2f} / ₹{target:.2f} ({progress_pct:.1f}%)\n"
                f"Remaining: ₹{remaining:.2f}\n"
                f"Days left: {days_left}"
            )
        finally:
            session.close()

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


@tool
def analyze_spending_trend(user_id: int, days: int = 30) -> str:
    """Analyze spending trends over a period."""
    session = get_db().get_session()
    try:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        query = select(
            Transaction.category,
            func.sum(Transaction.amount).label("total")
        ).where(
            Transaction.user_id == user_id,
            Transaction.transaction_type == TransactionType.debit,
            Transaction.transaction_date >= since
        ).group_by(Transaction.category)
        
        results = session.execute(query).all()
        
        if not results:
            return "No spending data found"
        
        trend = "\n".join([f"  {r.category}: ₹{float(r.total):.2f}" for r in results])
        return f"Spending by category (last {days} days):\n{trend}"
    finally:
        session.close()

# ============================================================================
# STATE GRAPH SETUP (Reference)
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def create_stateful_agent(user_id: int, api_key: str = None):
    if not HAS_GEMINI or not (api_key or os.getenv("GEMINI_API_KEY")):
        print("⚠ GEMINI not available")
        return None
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    tools = [
        AggregateSpendByCategoryTool(user_id=user_id),
        ListCategoriesForUserTool(user_id=user_id),
        GoalProgressTool(user_id=user_id),
    ]
    llm_with_tools = llm.bind_tools(tools)
    
    def call_model(state):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    async def call_tools(state):
        messages = state["messages"]
        last_message = messages[-1]
        
        if not hasattr(last_message, 'tool_calls'):
            return {"messages": []}
        
        tool_calls = last_message.tool_calls
        results = []
        
        for tool_call in tool_calls:
            tool = next(t for t in tools if t.name == tool_call["name"])
            result = await tool._arun(tool_call["args"])
            results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
        
        return {"messages": results}
    
    def should_continue(state):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_conditional_edges("model", should_continue)
    graph.add_edge("tools", "model")
    graph.set_entry_point("model")
    
    return graph.compile()

# ============================================================================
# TOKEN LOGGING
# ============================================================================

class TokenLogger:
    def __init__(self, log_file: str = "token_logs.csv"):
        self.log_file = log_file
        self._init_log_file()
    
    def _init_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("timestamp,purpose,input_tokens,output_tokens,total_tokens,model\n")
    
    def log(self, purpose: str, input_tokens: int = 0, output_tokens: int = 0, model: str = "gemini-2.5-flash"):
        total = input_tokens + output_tokens
        timestamp = datetime.now(timezone.utc).isoformat()
        
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{purpose},{input_tokens},{output_tokens},{total},{model}\n")
        
        print(f"[Token Log] {purpose}: {input_tokens} in + {output_tokens} out = {total} total")
    
    def get_stats(self) -> dict:
        if not os.path.exists(self.log_file):
            return {}
        
        import csv
        stats = {"total_tokens": 0, "calls": 0, "avg_tokens": 0}
        
        try:
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats["total_tokens"] += int(row.get("total_tokens", 0))
                    stats["calls"] += 1
            
            if stats["calls"] > 0:
                stats["avg_tokens"] = stats["total_tokens"] / stats["calls"]
        except Exception as e:
            print(f"Error reading stats: {e}")
        
        return stats

_token_logger = TokenLogger()

def log_tokens(purpose: str, input_tokens: int = 0, output_tokens: int = 0):
    _token_logger.log(purpose, input_tokens, output_tokens)

def get_token_stats():
    return _token_logger.get_stats()

# ============================================================================
# FINISHED AGENT DEMO
# ============================================================================

class Agent:
    """A complete, working financial advisor agent for demonstration."""
    
    def __init__(self, user_id: int, api_key: str = None, model: str = "gemini-2.5-flash"):
        self.user_id = user_id
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.llm = None
        self.memory = ConversationBufferWindowMemory(k=5)
        
        if HAS_GEMINI and self.api_key:
            genai.configure(api_key=self.api_key)
            self.llm = ChatGoogleGenerativeAI(
                model=model, 
                google_api_key=self.api_key,
                temperature=0.7
            )
            print(f"✓ LLM initialized: {model}")
        else:
            print("⚠ GEMINI API key not found.")
    
    def _create_tools(self):
        return [
            AggregateSpendByCategoryTool(user_id=self.user_id),
            ListCategoriesForUserTool(user_id=self.user_id),
            GoalProgressTool(user_id=self.user_id),
            analyze_spending_trend,
        ]
    
    async def run_query_with_tools(self, query: str) -> str:
        if not self.llm:
            return "❌ LLM not configured. Please provide GEMINI_API_KEY."
        
        tools = self._create_tools()
        llm_with_tools = self.llm.bind_tools(tools)
        
        system_msg = SystemMessage(
            content="You are a financial advisor. Use available tools to analyze the user's finances."
        )
        
        user_msg = HumanMessage(content=query)
        messages = [system_msg, user_msg]
        
        response = llm_with_tools.invoke(messages)
        print(f"\n[Agent] LLM response (has tools: {hasattr(response, 'tool_calls') and bool(response.tool_calls)})")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                
                print(f"  → Calling tool: {tool_name} with {tool_input}")
                
                tool = next((t for t in tools if t.name == tool_name), None)
                if tool:
                    result = await tool._arun(tool_input) if hasattr(tool, '_arun') else tool._run(**tool_input)
                    tool_results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
                    print(f"  ← Tool result: {result[:100]}...")
            
            messages.append(response)
            messages.extend(tool_results)
            final_response = llm_with_tools.invoke(messages)
            return final_response.content
        else:
            return response.content
    
    def run_query(self, query: str) -> str:
        return asyncio.run(self.run_query_with_tools(query))

def demo():
    """Demo the finished agent at the start of the session."""
    print("\n" + "=" * 70)
    print("DEMO: Complete Financial Advisor Agent")
    print("=" * 70)
    
    if not HAS_GEMINI or not os.getenv("GEMINI_API_KEY"):
        print("⚠ GEMINI_API_KEY not set. Skipping demo.")
        return
    
    setup = Setup()
    user_id = setup.get_user_id()
    
    agent = Agent(user_id, api_key=os.getenv("GEMINI_API_KEY"))
    
    print("\n📋 This is what we'll build together in this session:")
    print("   - Custom tools for financial analysis")
    print("   - An agent that calls tools based on natural language")
    print("   - Multi-turn conversations with memory")
    print("\n🚀 Watch it in action:\n")
    
    query = "What's my food spending this month and show me my goal progress?"
    print(f"User: {query}\n")
    
    try:
        result = agent.run_query(query)
        print(f"\nAgent Response:\n{result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70 + "\n")

# ============================================================================
# EXPORTS FOR EASY IMPORTING
# ============================================================================

__all__ = [
    "Setup",
    "Agent",
    "AggregateSpendByCategoryTool",
    "ListCategoriesForUserTool",
    "GoalProgressTool",
    "analyze_spending_trend",
    "AgentState",
    "create_stateful_agent",
    "TokenLogger",
    "log_tokens",
    "get_token_stats",
    "get_db",
    "demo",
]

if __name__ == "__main__":
    print("LangChain Financial Advisor - Quick Start Module")
    print("=" * 70)
    print("\nImport Examples:")
    print("\n1. Local/Colab - Run demo:")
    print("   python langchain_quickstart.py")
    print("\n2. Import and use immediately:")
    print("   from langchain_quickstart import Setup, Agent")
    print("   setup = Setup()")
    print("   agent = Agent(setup.get_user_id())")
    print("   response = agent.run_query('What is my spending?')")
    print("\n3. Import tools for building your own agent:")
    print("   from langchain_quickstart import (")
    print("       Setup, AggregateSpendByCategoryTool, ListCategoriesForUserTool")
    print("   )")
    print("   setup = Setup()")
    print("   # Build your agent step by step...")
    print("\n" + "=" * 70)
    
    demo()
