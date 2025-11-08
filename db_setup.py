"""
Minimal database setup for financial advisor.
Just import and call setup() to get started.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Numeric, 
    Enum, DateTime, ForeignKey, func, select
)
from sqlalchemy.orm import relationship, Session, sessionmaker, declarative_base
import enum

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


class DB:
    """Database interface."""
    
    def __init__(self, db_path: str = "financial_advisor.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        return self.SessionLocal()
    
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
                       category: str, description: str = "", days_ago: int = 0) -> Transaction:
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
                 current_amount: float = 0, category: str = None, days_ahead: int = 365) -> Goal:
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


# Global DB instance
_db: Optional[DB] = None

def setup(db_path: str = "financial_advisor.db") -> DB:
    """Initialize and return the database."""
    global _db
    _db = DB(db_path)
    print(f"✓ Database initialized at {db_path}")
    return _db

def get_db() -> DB:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = DB()
    return _db

import random
from faker import Faker

def populate_dummy_data(db: DB, num_users: int = 5, num_transactions: int = 40, num_goals: int = 5):
    """
    Populates the database with a specified number of dummy users,
    transactions, and goals.
    """
    
    fake = Faker()
    
    # Define some realistic categories
    debit_categories = ['Groceries', 'Rent', 'Utilities', 'Dining', 'Transport', 'Entertainment', 'Shopping', 'Healthcare']
    credit_categories = ['Salary', 'Freelance', 'Bonus', 'Gift', 'Investment']
    goal_categories = ['Vacation', 'New Car', 'Emergency Fund', 'Down Payment', 'Retirement', 'Gadget']
    
    print(f"Populating database with {num_users} users, {num_transactions} transactions, and {num_goals} goals...")
    
    user_ids = []
    try:
        # --- 1. Create Users ---
        for _ in range(num_users):
            try:
                # Use faker to generate unique-ish names and emails
                user = db.add_user(
                    username=fake.user_name(),
                    email=fake.email()
                )
                user_ids.append(user.id)
            except Exception as e:
                # This catches errors if faker generates a non-unique username/email
                print(f"Could not add user: {e}")
        
        if not user_ids:
            print("No users were created. Aborting data population.")
            return

        # --- 2. Create Transactions ---
        for _ in range(num_transactions):
            tx_type = random.choice(['credit', 'debit'])
            
            if tx_type == 'credit':
                category = random.choice(credit_categories)
                # Credits are usually larger amounts
                amount = round(random.uniform(500.0, 7000.0), 2)
            else:
                category = random.choice(debit_categories)
                # Debits are usually smaller
                amount = round(random.uniform(5.0, 500.0), 2)
            
            db.add_transaction(
                user_id=random.choice(user_ids),
                amount=amount,
                transaction_type=tx_type,
                category=category,
                description=fake.sentence(nb_words=4),
                days_ago=random.randint(0, 365) # Transactions from the past year
            )
        
        # --- 3. Create Goals ---
        for _ in range(num_goals):
            target = round(random.uniform(1000.0, 25000.0), 2)
            # Start with some progress on the goal
            current = round(random.uniform(0.0, target * 0.8), 2) 
            
            db.add_goal(
                user_id=random.choice(user_ids),
                title=f"Save for {random.choice(goal_categories)}",
                target_amount=target,
                current_amount=current,
                category=random.choice(goal_categories),
                days_ahead=random.randint(60, 730) # Goals due in the next 2 months to 2 years
            )

        print("✓ Dummy data populated successfully.")

    except Exception as e:
        print(f"An error occurred during data population: {e}")


if __name__ == "__main__":
    # This block demonstrates how to use the functions
    
    # 1. Setup the database
    # Using "test_advisor.db" will create a file.
    # Using "sqlite:///:memory:" will create an in-memory DB for testing.
    db_instance = setup(db_path="rag.db")
    
    # 2. Populate with dummy data
    # (5 users + 40 transactions + 5 goals = 50 total entries)
    populate_dummy_data(db_instance, num_users=5, num_transactions=40, num_goals=5)
    
    # 3. (Optional) Verify the data was added
    session = db_instance.get_session()
    try:
        user_count = session.query(func.count(User.id)).scalar()
        tx_count = session.query(func.count(Transaction.id)).scalar()
        goal_count = session.query(func.count(Goal.id)).scalar()
        
        print("\n--- Database Verification ---")
        print(f"Total Users: {user_count}")
        print(f"Total Transactions: {tx_count}")
        print(f"Total Goals: {goal_count}")
        
        # Show transactions for the first user
        first_user = session.scalars(select(User)).first()
        if first_user:
            print(f"\nSample data for user: {first_user.username}")
            for tx in first_user.transactions:
                print(f"  - TX: {tx.transaction_type.name} {tx.amount} on {tx.transaction_date.date()} for {tx.category}")
            for goal in first_user.goals:
                print(f"  - GOAL: {goal.title} ({goal.current_amount} / {goal.target_amount})")
    finally:
        session.close()