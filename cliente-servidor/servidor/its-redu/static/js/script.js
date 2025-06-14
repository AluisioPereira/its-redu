
// Seleciona o campo de entrada
var messageInput = document.getElementById('message-input');

// Seleciona o botão de enviar
var sendButton = document.getElementById('send-button');

// Adiciona um evento de escuta de entrada ao campo de entrada de texto
messageInput.addEventListener('input', function() {
    // Habilita ou desabilita o botão com base no comprimento do texto no campo de entrada
    sendButton.disabled = messageInput.value.trim() === '';
});

// Adiciona um evento de escuta de teclado ao campo de entrada de texto
messageInput.addEventListener('keyup', function(event) {
    // Verifica se a tecla pressionada é Enter
    if (event.key === 'Enter') {
        // Envia a mensagem se a tecla Enter for pressionada
        sendMessage();
    }
});

// Adiciona um evento de escuta de clique ao botão de enviar
sendButton.addEventListener('click', function() {
    // Envia a mensagem quando o botão de enviar for clicado
    sendMessage();
});

function sendMessage() {
  var messageInput = document.getElementById("message-input");
  var message = messageInput.value;
  messageInput.value = "";

  var chatContainer = document.getElementById("chat-container");

  // Criar balão de mensagem do usuário (humano)
  var senderBubble = document.createElement("div");
  senderBubble.className = "message sender";

  
  var senderImg = document.createElement("div");
  senderImg.className = "sender-img";
  var img = document.createElement("img");
  img.src = "/static/img/humano.png"; // Caminho para a imagem do humano
  img.alt = "Human";
  senderImg.appendChild(img);
  senderBubble.appendChild(senderImg);


  var bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message;
  senderBubble.appendChild(bubble);

  chatContainer.appendChild(senderBubble);

  // Enviar a mensagem para o servidor Flask e obter a resposta
  fetch("/send_message", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message: message }),
  })
    .then((response) => response.json())
    .then((data) => {
      // Criar balão de mensagem do robô
      var receiverBubble = document.createElement("div");
      receiverBubble.className = "message receiver";


        var senderImg = document.createElement('div');
        senderImg.className = 'sender-img';
        var img = document.createElement('img');
        img.src = '/static/img/robo.png'; // Caminho para a imagem do robô
        img.alt = 'Robot';
        senderImg.appendChild(img);
        receiverBubble.appendChild(senderImg);
       


      var bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = data.message;
      receiverBubble.appendChild(bubble);

      chatContainer.appendChild(receiverBubble);

      // Rolar para o final do chat
      chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}
