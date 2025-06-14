<template>
    <ion-menu content-id="main-content" type="overlay">
      <ion-content>
        <ion-img id="logo-its-redu" src="/its-redu2.png" alt="Its.Redu"></ion-img>
        <ion-list id="interacao-list">
          <ion-menu-toggle :auto-hide="false" v-for="(page, index) in appPages" :key="index">
            <ion-item
              @click="selectPage(index)"
              router-direction="root"
              :router-link="page.url"
              lines="none"
              :detail="false"
              class="hydrated"
              :class="{ selected: selectedIndex === index }"
            >
              <ion-icon aria-hidden="true" slot="start" :ios="page.iosIcon" :md="page.mdIcon"></ion-icon>
              <ion-label>{{ page.title }}</ion-label>
            </ion-item>
          </ion-menu-toggle>
        </ion-list>
        <ion-list id="labels-list">
          <ion-list-header>Próximas Tutorias</ion-list-header>
          <ion-item v-for="(label, index) in labels" :key="index" lines="none">
            <ion-icon aria-hidden="true" slot="start" :ios="bookmarkOutline" :md="bookmarkSharp"></ion-icon>
            <ion-label>{{ label }}</ion-label>
          </ion-item>
          
        </ion-list>
      </ion-content>
    </ion-menu>
  </template>
  
  <script setup>
  import { ref } from 'vue';
  import {
    IonContent,
    IonIcon,
    IonItem,
    IonLabel,
    IonList,
    IonListHeader,
    IonMenu,
    IonMenuToggle,
  } from '@ionic/vue';
  import {
    bookmarkOutline,
    bookmarkSharp,
    peopleSharp,
    pieChartSharp,
    heartOutline,
    heartSharp,
    helpCircleSharp,
    chatboxEllipses,
    schoolSharp,
  } from 'ionicons/icons';
  
  const selectedIndex = ref(0);
  const appPages = [
    {
      title: 'Interações',
      url: '/its/interacao',
      iosIcon: peopleSharp,
      mdIcon: peopleSharp,
    },
    {
      title: 'Desempenho',
      url: '/its/desempenho',
      iosIcon: pieChartSharp,
      mdIcon: pieChartSharp,
    },
    {
      title: 'Busca Ativa',
      url: '/its/busca-ativa',
      iosIcon: heartOutline,
      mdIcon: heartSharp,
    },
    {
      title: 'Ajuda do Redu',
      url: '/its/ajuda-redu',
      iosIcon: helpCircleSharp,
      mdIcon: helpCircleSharp,
    },
    {
      title: 'Chat.Its',
      url: '/its/chat-its',
      iosIcon: chatboxEllipses,
      mdIcon: chatboxEllipses,
    },
    {
      title: 'Tutores',
      url: '/its/tutores',
      iosIcon: schoolSharp,
      mdIcon: schoolSharp,
    },
  ];
  
  const labels = ['Design thinking', 'Inovação', 'Mudando o jogo'];
  
  const path = window.location.pathname.split('its/')[1];
  if (path !== undefined) {
    selectedIndex.value = appPages.findIndex((page) => page.title.toLowerCase() === path.toLowerCase());
  }
  
  const selectPage = (index) => {
    selectedIndex.value = index;
  };
  </script>
  
  <style scoped>
  #logo-its-redu {
    width: 40%;
    right: 30%;
  }
  
  ion-menu {
    --background: var(--ion-item-background, var(--ion-background-color, #fff));
  }
  
  ion-menu md {
    --padding-start: 8px;
    --padding-end: 8px;
    --padding-top: 20px;
    --padding-bottom: 20px;
  }
  
  ion-menu md ion-list {
    padding: 20px 0;
  }
  
  ion-menu md ion-note {
    margin-bottom: 30px;
  }
  
  ion-menu md ion-list-header,
  ion-menu md ion-note {
    padding-left: 10px;
  }
  
  ion-menu ios {
    --padding-bottom: 20px;
  }
  
  ion-menu ios ion-list {
    padding: 20px 0 0 0;
  }
  
  ion-menu ios ion-note {
    line-height: 24px;
    margin-bottom: 20px;
  }
  
  ion-menu ios ion-item {
    --padding-start: 16px;
    --padding-end: 16px;
    --min-height: 50px;
  }
  
  ion-menu ios ion-item.selected ion-icon {
    color: var(--ion-color-primary);
  }
  
  ion-menu ios ion-item ion-icon {
    font-size: 24px;
    color: #73849a;
  }
  
  ion-menu ios ion-list#labels-list ion-list-header {
    margin-bottom: 8px;
  }
  
  ion-menu ios ion-list-header,
  ion-menu ios ion-note {
    padding-left: 16px;
    padding-right: 16px;
  }
  
  ion-menu ios ion-note {
    margin-bottom: 8px;
  }
  
  ion-note {
    display: inline-block;
    font-size: 16px;
    color: var(--ion-color-medium-shade);
  }
  
  ion-item.selected {
    --background: rgba(var(--ion-color-primary-rgb), 0.14);
  }
  
  ion-menu md ion-item.selected {
    --background: rgba(var(--ion-color-primary-rgb), 0.14);
  }
  
  ion-menu ios ion-item.selected {
    --background: rgba(var(--ion-color-primary-rgb), 0.14);
  }
  
  ion-item.selected ion-icon {
    color: var(--ion-color-primary);
  }
  
</style>
