# routes/auth_routes.py
from flask import Blueprint, request, redirect, url_for, flash
from flask_login import login_user, logout_user, current_user
from werkzeug.security import check_password_hash
from flask import Flask, render_template, redirect, url_for
from flask_login import LoginManager, current_user
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_user, logout_user, current_user, login_required
from ..models.local_user import LocalUser
import app

auth_bp = Blueprint('auth', __name__, template_folder='templates/auth')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        user = LocalUser.get_by_email(email)
        if not user or not user.check_password(password):
            flash('Invalid email or password', 'error')
            return redirect(url_for('auth.login'))
        
        login_user(user, remember=remember)
        return redirect(url_for('index'))
    
    return render_template('auth/login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

@auth_bp.route('/check')
def auth_check():
    return jsonify(authenticated=current_user.is_authenticated)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Local development
        if current_app.config['ENV'] == 'development':
            from ..models.local_user import LocalUser
            from werkzeug.security import generate_password_hash
            
            if email in current_app.users:
                flash('Email already registered', 'danger')
            else:
                user_id = max(u.id for u in current_app.users.values()) + 1
                current_app.users[email] = LocalUser(
                    id=user_id,
                    email=email,
                    password=generate_password_hash(password)
                )
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('auth.login'))
        
        # Production with AWS Cognito
        else:
            try:
                # Register in Cognito
                response = current_app.cognito.sign_up(
                    ClientId=current_app.config['COGNITO_CLIENT_ID'],
                    Username=email,
                    Password=password,
                    UserAttributes=[
                        {'Name': 'email', 'Value': email}
                    ]
                )
                
                # Store in DynamoDB
                from ..models.dynamodb_user import DynamoDBUser
                user = DynamoDBUser.create(
                    email=email,
                    dynamodb=current_app.dynamodb
                )
                
                flash('Registration successful! Please check your email to confirm.', 'success')
                return redirect(url_for('auth.login'))
            
            except current_app.cognito.exceptions.UsernameExistsException:
                flash('Email already registered', 'danger')
    
    return render_template('auth/register.html')