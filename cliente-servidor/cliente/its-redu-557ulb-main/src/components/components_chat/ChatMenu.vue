<template>
  <ion-tabs>
    <ion-router-outlet></ion-router-outlet>
    <ion-tab-bar slot="top">
      <ion-tab-button tab="ajuda" href="/its/chat-its/ajuda">
        <ion-icon aria-hidden="true" :ios="helpCircleSharp" :md="helpCircleSharp"></ion-icon>
        <ion-label>Ajuda Redu</ion-label>
      </ion-tab-button>

      <ion-tab-button tab="chat" href="/its/chat-its/chat">
        <ion-icon aria-hidden="true" :ios="chatboxEllipses" :md="chatboxEllipses"></ion-icon>
        <ion-label>Chat.Its</ion-label>
      </ion-tab-button>

      <ion-tab-button tab="tutores" href="/its/chat-its/tutores">
        <ion-icon aria-hidden="true" :ios="schoolSharp" :md="schoolSharp"></ion-icon>
        <ion-label>Tutores</ion-label>
      </ion-tab-button>
    </ion-tab-bar>
  </ion-tabs>
</template>

<script setup>
import { IonTabBar, IonTabButton, IonTabs, IonLabel, IonIcon, IonRouterOutlet } from '@ionic/vue';
import { chatboxEllipses, helpCircleSharp, schoolSharp } from 'ionicons/icons';

</script>