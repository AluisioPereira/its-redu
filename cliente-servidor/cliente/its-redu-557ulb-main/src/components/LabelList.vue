<template>
    <ion-list id="labels-list">
      <ion-list-header>Próximas Tutorias</ion-list-header>
      <ion-item v-for="(label, index) in labels" :key="index" lines="none">
        <ion-icon aria-hidden="true" slot="start" :ios="bookmarkOutline" :md="bookmarkSharp"></ion-icon>
        <ion-label>{{ label }}</ion-label>
      </ion-item>
      <ion-note></ion-note>
      <ion-note></ion-note>
      <ion-list-header>
        <p style="color: rgb(97, 97, 252)">I</p>
        <p style="color: rgb(253, 172, 51)">t</p>
        <p style="color: rgb(3, 167, 39)">s</p>
        <p style="color: rgb(93, 93, 97)">.</p>
        <p style="color: rgb(3, 3, 219)">Redu</p>
      </ion-list-header>
    </ion-list>
  </template>
  
  <script>
  import { bookmarkOutline, bookmarkSharp } from 'ionicons/icons';
  
  export default {
    name: 'LabelList',
    props: {
      labels: {
        type: Array,
        required: true,
      },
    },
    data() {
      return {
        bookmarkOutline,
        bookmarkSharp,
      };
    },
  };
  </script>
  
  <style scoped>
  ion-list-header {
    font-size: 16px;
    margin-bottom: 18px;
    color: #757575;
    min-height: 26px;
  }
  
  ion-note {
    display: inline-block;
    font-size: 16px;
    color: var(--ion-color-medium-shade);
  }
  </style>