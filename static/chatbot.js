class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chat_button'),
            chatBox: document.querySelector('.support'),
            sendButton: document.querySelector('.send_button'),
            micButton: document.querySelector('.mic_button')
        }
        this.state = false;
        this.messages = [];
    }

    display() {
        const { openButton, chatBox, sendButton, micButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));
        micButton.addEventListener('click', () => this.startRecording(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
            let msg2 = { name: "Agribot", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox)
            textField.value = ''

        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
          });
    }

    startRecording(chatbox) {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'te-IN'; // Set language to telugu
        recognition.start();

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            console.log('You said:', transcript);

            // Send the transcript to the server for translation
            fetch('http://127.0.0.1:5000/translate', {
                method: 'POST',
                body: JSON.stringify({ message: transcript, target_lang: 'en' }), // 'en' is english
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json'
                },
            })
            .then(response => response.json())
            .then(data => {
                const translatedText = data.translated_text;
    
                // Display the translated text in the input field
                const inputField = chatbox.querySelector('input');
                inputField.value = translatedText;
                inputField.focus(); // Optionally focus the input field
    
                // Display the original recorded voice in the chatbox
                const msg1 = { name: "User", message: transcript };
                this.messages.push(msg1);
                this.updateChatText(chatbox);
            })
            .catch((error) => {
                console.error('Translation error:', error);
            });
        };

        recognition.onerror = (event) => {
            console.error('Error occurred in recognition: ' + event.error);
        };
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "User")
            {
                html += '<div class="messages_item messages_item_student">' + item.message + '</div>'
            }
            else
            {
                html += '<div class="messages_item messages_item_bot">' + item.message + '</div>'
            }
          });

        const chatmessage = chatbox.querySelector('.messages');
        chatmessage.innerHTML = html;
    }
}


const chatbox = new Chatbox();
chatbox.display();