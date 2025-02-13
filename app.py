import os
import sqlite3
from flask import Flask, flash, render_template, redirect, session, url_for, g
from flask.globals import request
from werkzeug.utils import secure_filename
from workers import pdf2text, txt2questions
from flask_mail import Mail, Message
from flask import jsonify


UPLOAD_FOLDER = './pdf/'
currentLocation = os.path.dirname(os.path.abspath(__file__))
questionLength = 0
correctAnswers = []
questions = dict()
app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.elasticemail.com'
app.config['MAIL_PORT'] = 2525
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'sujayhdeshpande@gmail.com'
app.config['MAIL_PASSWORD'] = '01FD97033ABC4389DED96CD0237987AA6FDF'
app.config['MAIL_DEFAULT_SENDER'] = 'sujayhdeshpande@gmail.com'


mail = Mail(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'TeamQuizifyr'


@ app.route('/')
def index():
    '''First page'''
    return render_template('index.html')

@ app.route('/upload')
def upload():
    if g.user:
        return render_template('upload.html')
    return render_template('unauthorized.html')

@app.before_request
def before_request():
    g.user = None
    if 'user' in session:
        g.user = session['user']
 
@app.route('/dropsession')
def dropsession():
    session.pop('user', None)
    return redirect('/')

@ app.route('/signin')
def signin():
    return render_template('signin.html')


@app.route('/signin', methods=['POST'])
def checklogin():
    UN = request.form['Username']
    PW = request.form['Password']

    sqlconnection = sqlite3.Connection(currentLocation+"\\Login.db")
    cursor = sqlconnection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS Users(Username text, Password text);")
    sqlconnection.commit()
    query1 = "SELECT Username, Password from Users WHERE Username = '{un}' AND Password = '{pw}'".format(un=UN, pw=PW)
    
    rows = cursor.execute(query1)
    rows = rows.fetchall()
    if len(rows) == 1:
        if request.method == 'POST':
            session.pop('user', None)

            if request.form['Password'] == PW:
                session['user'] = request.form['Username']
                if session['user'] == 'sujaydeshpande2406@gmail.com':
                    return redirect('/admin')
                else:
                    return redirect('/upload')  
    else:
        flash("❌ Invalid credentials! Try again or sign up!")
    return render_template('signin.html')

@app.route('/admin')
def admin():
    if 'user' in session and session['user'] == 'sujaydeshpande2406@gmail.com':
        logged_in_users = get_logged_in_users_count()  
        quiz_count = get_quiz_count()
        return render_template('admin.html', logged_in_users=logged_in_users, quiz_count=quiz_count)
    else:
        return redirect('/signin') 
    
def get_quiz_count():
    try:
        sqlconnection = sqlite3.Connection(currentLocation + "\\count.db")
        cursor = sqlconnection.cursor()
        cursor.execute("SELECT count FROM QuizCount")
        count = cursor.fetchone()
        sqlconnection.close()
        if count:
            return count[0]
        else:
            return 0
    except Exception as e:
        print("Error fetching quiz count:", e)
        return 0


def get_logged_in_users_count():
    try:
        sqlconnection = sqlite3.Connection(currentLocation + "\\Login.db")
        cursor = sqlconnection.cursor()
        query = "SELECT COUNT(*) FROM Users"
        cursor.execute(query)
        count = cursor.fetchone()[0]
        sqlconnection.close()
        return count
    except Exception as e:
        print("Error fetching logged-in users count:", e)
        return 0
    

@app.route('/admin/action', methods=['POST'])
def admin_action():
    pass  



@ app.route('/signup', methods=['GET', 'POST'])
def registerpage():
    if request.method == "POST":
        dUN = request.form['DUsername']
        dPW = request.form['DPassword']
        cPW = request.form['confPw']
        
        sqlconnection = sqlite3.Connection(currentLocation + "\\Login.db")
        cursor = sqlconnection.cursor()
        
        try:
            if cPW == dPW:
                checkUser = "SELECT Username, Password from Users WHERE Username = ?"
                cursor.execute(checkUser, (dUN,))
                row = cursor.fetchone()
                
                if row:
                    flash("❗ Email ID already exists, try logging in!")
                else:
                    query1 = "INSERT INTO Users VALUES (?, ?)"
                    cursor.execute(query1, (dUN, dPW))
                    sqlconnection.commit()
                    
                    flash("🎉 Awesome! Account created, now log in to your account!")
                    return redirect('/signin')
            else:
                flash("🤨 Password and Confirm password don't match, try again!")
                return render_template('signup.html')
        except Exception as e:
            flash("⛔ Unknown error occurred, try again later!")
            print(e)  
            return render_template('signup.html')
    
    return render_template('signup.html')

@ app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    global questions
    sqlconnection = sqlite3.Connection(currentLocation + "\\count.db")
    cursor = sqlconnection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS QuizCount(count INTEGER)")
    sqlconnection.commit()

    cursor.execute("SELECT * FROM QuizCount")
    record = cursor.fetchone()

    if record is None:
        cursor.execute("INSERT INTO QuizCount (count) VALUES (1)")
    else:
        cursor.execute("UPDATE QuizCount SET count = count + 1")
    
    sqlconnection.commit()
    
    UPLOAD_STATUS = False
    questions = dict()
    if not os.path.isdir('./pdf'):
        os.mkdir('./pdf')

    if request.method == 'POST':
        try:
            uploaded_file = request.files['file']
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                secure_filename(uploaded_file.filename))
            file_exten = uploaded_file.filename.rsplit('.', 1)[1].lower()

            uploaded_file.save(file_path)
            uploaded_content = pdf2text(file_path, file_exten)
            questions = txt2questions(uploaded_content)

            global questionLength
            questionLength = len(questions)

            global correctAnswers
            for i in range(questionLength):
                correctAnswers.append(questions[i+1]['answer'])

            if uploaded_content is not None:
                UPLOAD_STATUS = True
        except Exception as e:
            print(e)
    return render_template(
        'quiz.html',
        uploaded=UPLOAD_STATUS,
        questions=questions,
        size=len(questions))



@ app.route('/result', methods=['POST', 'GET'])
def result():
    if g.user:
        global questions
        correct_q = 0
        selectedOptions = [] 
 
 
        for i in range(questionLength):
            radioGroupName = 'question' + str(i+1)
            option = request.form.getlist(radioGroupName)
            if option:  
                
                selectedOptions.append(option[0])
            else:
                pass

        
        for i in selectedOptions:
            for k in correctAnswers:
                if(i == k):
                    correct_q += 1    

        
        correct_mcqs = []
        incorrect_mcqs = []



        
        for i in range(questionLength):
            question_number = i + 1
            correct_answer = correctAnswers[i]
            user_answer = selectedOptions[i]
            question = questions[question_number]['question']
            options = questions[question_number]['options']

            
            is_correct = correct_answer == user_answer

            
            mcq = {
                'question_number': question_number,
                'question': question,
                'options': options,
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct
            }

            
            
            if is_correct:
                correct_mcqs.append(mcq)
            else:
                incorrect_mcqs.append(mcq)

        
        total = len(correct_mcqs) + len(incorrect_mcqs)

       
        result_html = render_template('result.html', total=total, correct=correct_q,
                                      correct_mcqs=correct_mcqs, incorrect_mcqs=incorrect_mcqs)
        
        # sEnd emailll
        send_email(g.user, result_html)

        return result_html

    return render_template('unauthorized.html')




def send_email(recipient, html_content):
    msg = Message('Quiz Result: Quizifer', sender='sujayhdeshpande@gmail.com', recipients=[recipient])
    msg.html = html_content

    try:
        mail.send(msg)
        return 'Email sent successfully!'
    except Exception as e:
        return f'Failed to send email: {str(e)}'


@app.route('/admin/users', methods=['GET'])
def get_users():
    try:
        conn = sqlite3.connect(currentLocation + "\\Login.db")
        cursor = conn.cursor()

        cursor.execute("SELECT Username FROM Users")
        users = cursor.fetchall()

        user_list = []
        for user in users:
            user_dict = {
                'Username': user[0],
            }
            user_list.append(user_dict)
        conn.close()
        return jsonify({'users': user_list})

    except Exception as e:
        print("Error fetching users:", e)
        return jsonify({'error': 'Failed to fetch users'}), 500

@ app.route('/send_email', methods=['POST'])
def send_custom_email():
    if g.user:
        
        recipient_email = request.form.get('recipient_email')

        email_html = render_template('email_template.html')

        send_email(recipient_email, email_html)
        
        flash('Email sent successfully!')
        return redirect(url_for('index'))

    return render_template('unauthorized.html')



@ app.errorhandler(404)
def invalid_route(e):
    return render_template('404.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
