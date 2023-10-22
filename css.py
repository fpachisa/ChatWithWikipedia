css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #fca895   
}
.chat-message.bot {
    background-color: #b8b6b4
}
.chat-message .avatar {
  width: 10%;
}
.chat-message .avatar img {
  max-width: 58px;
  max-height: 58px;
  border-radius: 50%;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #000;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src= "https://i.ibb.co/ZLmfqfp/chatbot.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/dc0h7Yh/user.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
