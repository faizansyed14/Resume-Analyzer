# models/dynamodb_user.py
import uuid
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
def ensure_user_table_exists(dynamodb):
    try:
        table = dynamodb.create_table(
            TableName='AlphaDataUsers',
            KeySchema=[
                {'AttributeName': 'user_id', 'KeyType': 'HASH'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'user_id', 'AttributeType': 'S'},
                {'AttributeName': 'email', 'AttributeType': 'S'}
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'EmailIndex',
                    'KeySchema': [
                        {'AttributeName': 'email', 'KeyType': 'HASH'}
                    ],
                    'Projection': {
                        'ProjectionType': 'ALL'
                    },
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        table.wait_until_exists()
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        pass

class DynamoDBUser(UserMixin):
    def __init__(self, user_id, email, password_hash=None):
        self.user_id = user_id
        self.email = email
        self.password_hash = password_hash
    
    def get_id(self):
        return self.user_id
    
    @classmethod
    def get(cls, user_id, dynamodb):
        table = dynamodb.Table('AlphaDataUsers')
        response = table.get_item(Key={'user_id': user_id})
        if 'Item' in response:
            item = response['Item']
            return cls(
                user_id=item['user_id'],
                email=item['email'],
                password_hash=item.get('password_hash')
            )
        return None
    
    @classmethod
    def get_by_email(cls, email, dynamodb):
        table = dynamodb.Table('AlphaDataUsers')
        response = table.query(
            IndexName='EmailIndex',
            KeyConditionExpression='email = :email',
            ExpressionAttributeValues={':email': email}
        )
        if response['Items']:
            item = response['Items'][0]
            return cls(
                user_id=item['user_id'],
                email=item['email'],
                password_hash=item.get('password_hash')
            )
        return None
    
    @classmethod
    def create(cls, email, dynamodb):
        table = dynamodb.Table('AlphaDataUsers')
        user_id = str(uuid.uuid4())
        table.put_item(
            Item={
                'user_id': user_id,
                'email': email,
                'created_at': datetime.utcnow().isoformat()
            }
        )
        return cls(user_id, email)
    
    def set_password(self, password, dynamodb):
        self.password_hash = generate_password_hash(password)
        table = dynamodb.Table('AlphaDataUsers')
        table.update_item(
            Key={'user_id': self.user_id},
            UpdateExpression='SET password_hash = :ph',
            ExpressionAttributeValues={':ph': self.password_hash}
        )
    
    def check_password(self, password):
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)