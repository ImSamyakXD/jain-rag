const input = document.getElementById("input");
const sendButton = document.getElementById("send");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const newChat = document.getElementById("newChat");


// =========================
// SEND BUTTON
// =========================

sendButton.addEventListener("click", sendMessage);


// =========================
// SEND MESSAGE
// =========================

async function sendMessage() {

    const question = input.value.trim();

    if (!question) {
        return;
    }


    // Hide welcome screen
    welcome.style.display = "none";


    // Show user question
    addMessage(question, "user");


    // Clear input
    input.value = "";


    // Loading message
    const loading = addMessage(
        "Thinking...",
        "bot"
    );


    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);

        const response = await fetch(
            "/chat",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                signal: controller.signal,
                body: JSON.stringify({
                    message: question
                })
            }
        );
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error("Server error: " + response.status);
        }

        const data = await response.json();

        // Remove loading
        loading.remove();

        // Show answer
        addMessage(
            data.answer,
            "bot"
        );

    }

    catch (error) {

        console.error(error);

        loading.remove();

        addMessage(
            "⏳ Server is warming up or connecting. Please wait a few seconds and send your question again!",
            "bot"
        );

    }

}


// =========================
// ADD MESSAGE
// =========================

function addMessage(text, type) {

    const wrapper = document.createElement("div");

    wrapper.className = "message " + type;


    const bubble = document.createElement("div");


    if (type === "user") {

        bubble.className = "user-bubble";

        bubble.textContent = text;

    }

    else {

        bubble.className = "bot-bubble";

        bubble.innerHTML = `
            <div class="bot-title">
                JainGPT
            </div>

            <div class="answer-content">
                ${formatAnswer(text)}
            </div>
        `;
    }


    wrapper.appendChild(bubble);

    messages.appendChild(wrapper);

    messages.scrollTop = messages.scrollHeight;

    return wrapper;
}

function formatAnswer(text) {

    // Escape HTML first
    text = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");


    // Bold: **text**
    text = text.replace(
        /\*\*(.*?)\*\*/g,
        "<strong>$1</strong>"
    );


    // Headings: ## Heading
    text = text.replace(
        /^### (.*)$/gm,
        "<h4>$1</h4>"
    );

    text = text.replace(
        /^## (.*)$/gm,
        "<h3>$1</h3>"
    );

    text = text.replace(
        /^# (.*)$/gm,
        "<h2>$1</h2>"
    );


    // Numbered lists
    text = text.replace(
        /^\s*(\d+)\.\s+(.*)$/gm,
        "<div class='list-item'><b>$1.</b> $2</div>"
    );


    // Bullet lists
    text = text.replace(
        /^\s*[-*]\s+(.*)$/gm,
        "<div class='list-item'>• $1</div>"
    );


    // Horizontal line
    text = text.replace(
        /^---$/gm,
        "<hr>"
    );


    // Line breaks
    text = text.replace(
        /\n/g,
        "<br>"
    );


    return text;
}


// =========================
// QUICK QUESTIONS
// =========================

document
    .querySelectorAll("[data-question]")
    .forEach(button => {

        button.addEventListener(
            "click",
            function() {

                input.value =
                    this.dataset.question;

                sendMessage();

            }
        );

    });


// =========================
// ENTER KEY
// =========================

input.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();

        }

    }
);


// =========================
// NEW CHAT
// =========================

newChat.addEventListener(
    "click",
    function() {

        messages.innerHTML = "";

        welcome.style.display =
            "block";

        input.value = "";

    }
);