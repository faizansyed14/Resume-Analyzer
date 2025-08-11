from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import logging

logger = logging.getLogger(__name__)

class LocalUser(UserMixin):
    # Class-level storage for users
    _users = {}
    
    def __init__(self, id, email, password=None):
        self.id = id
        self.email = email
        self._password = generate_password_hash(password) if password else None
    
    @property
    def password(self):
        return self._password
    
    @password.setter
    def password(self, value):
        self._password = generate_password_hash(value)
    
    def check_password(self, password):
        return check_password_hash(self._password, password)
    
    @classmethod
    def get(cls, user_id):
        return cls._users.get(str(user_id))
    
    @classmethod
    def get_by_email(cls, email):
        # the inner parens turn the generator into the first arg to next; the second arg is the default
        return next(
            (user for user in cls._users.values() if user.email == email),
            None
        )

    
    @classmethod
    def create(cls, email, password, silent=False):
        """Create a new user.
        
        Args:
            email: User email
            password: User password
            silent: If True, won't raise error if user exists (default: False)
        
        Returns:
            The created user or None if user exists and silent=True
        """
        existing_user = cls.get_by_email(email)
        if existing_user:
            if silent:
                logger.info(f"User {email} already exists, returning existing user")
                return existing_user
            raise ValueError(f"User {email} already exists")
        
        new_id = max((user.id for user in cls._users.values()), default=0) + 1
        user = cls(id=new_id, email=email, password=password)
        cls._users[str(new_id)] = user
        logger.info(f"Created new user {email} with ID {new_id}")
        return user
    
    @classmethod
    def ensure_user_exists(cls, email, password):
        """Ensure a user exists, creating if necessary."""
        try:
            return cls.create(email, password)
        except ValueError:
            return cls.get_by_email(email)