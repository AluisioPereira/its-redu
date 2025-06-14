<template>
  <ion-app>
    <ion-split-pane content-id="main-content">
      <AppMenu />
      <ion-router-outlet id="main-content"></ion-router-outlet>
    </ion-split-pane>
  </ion-app>
</template>

<script setup>
import { IonApp, IonRouterOutlet, IonSplitPane } from '@ionic/vue';
import AppMenu from './components/AppMenu.vue';
</script>



