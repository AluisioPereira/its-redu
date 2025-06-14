<template>
  <ion-page>
    <ion-header>
      <ion-toolbar>
        <ChatMenu />
      </ion-toolbar>
    </ion-header>
    <ion-content>
      <!-- Conteúdo da página de Chat-Its -->
      <div class="chat-container" ref="chatContainer"></div>
    </ion-content>
    <ion-footer>
      <ion-toolbar>
        <div class="chat-its" id="chat-its">
          <div class="input-container">
            <ion-input
              type="text"
              v-model="message"
              placeholder="Digite uma pergunta ou comando"
              @input="handleInput"
              @keyup.enter="sendMessage"
            ></ion-input>
            <ion-button :disabled="isSendButtonDisabled" @click="sendMessage">Enviar</ion-button>
          </div>
        </div>
      </ion-toolbar>
    </ion-footer>
  </ion-page>
</template>

<script setup>
import { ref } from 'vue';
import ChatMenu from '../components/components_chat/ChatMenu.vue';
import { IonFooter, IonToolbar, IonInput, IonButton, IonContent, IonPage, IonHeader } from '@ionic/vue';

const message = ref('');
const isSendButtonDisabled = ref(true);
const chatContainer = ref(null);

function handleInput() {
  isSendButtonDisabled.value = message.value.trim() === '';
}

async function sendMessage() {
  if (message.value.trim() === '') return;

  const messageContent = message.value;
  message.value = '';

  // Criar balão de mensagem do usuário (humano)
  const senderBubble = document.createElement('div');
  senderBubble.className = 'message sender';

  /*
  const senderImg = document.createElement('div');
  senderImg.className = 'sender-img';
  const img = document.createElement('img');
  img.src = '/humano.png'; // Caminho para a imagem do humano
  img.alt = 'Human';
  senderImg.appendChild(img);
  senderBubble.appendChild(senderImg);
  */

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = messageContent;
  senderBubble.appendChild(bubble);

  chatContainer.value.appendChild(senderBubble);


  // Enviar a mensagem para o servidor Flask e obter a resposta
  
  try {
    const response = await fetch('http://127.0.0.1:5000/send_message', { 
      // Certifique-se de usar a URL correta do seu servidor Flask
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: messageContent }),
    });

    const data = await response.json();
      
      // Criar balão de mensagem do robô
      const receiverBubble = document.createElement('div');
      receiverBubble.className = 'message receiver';
/*
      const receiverImg = document.createElement('div');
      receiverImg.className = 'sender-img';
      const img = document.createElement('img');
      img.src = '/robo.png'; // Caminho para a imagem do robô
      img.alt = 'Robot';
      receiverImg.appendChild(img);
      receiverBubble.appendChild(receiverImg);
*/
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      
      bubble.textContent = data.message;
      receiverBubble.appendChild(bubble);

      chatContainer.value.appendChild(receiverBubble);

      // Rolar para o final do chat
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight;
    } catch (error) {
    console.error('Erro ao enviar a mensagem:', error);
  }
}
</script>

<style scoped>
.chat-container {
    width: 75%;
    height: 75%;
    overflow-y: auto;
    border-radius: 10px;
    overflow: hidden;
    padding: 20px;
    border-color: #007bff;
}

.message.sender {
  /* Estilos para a mensagem do remetente */
}

.message.receiver {
  /* Estilos para a mensagem do receptor */
}

.sender-img img {
  /* Estilos para a imagem do remetente */
}

.receiver-img img {
  /* Estilos para a imagem do receptor */
}

.bubble {
  /* Estilos para os balões de mensagem */
}

.input-container {
  display: flex;
  align-items: center;
  padding: 10px;
}

ion-input {
  flex: 1;
  margin-right: 10px;
}

</style>