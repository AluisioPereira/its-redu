import { createRouter, createWebHistory } from '@ionic/vue-router';

import InteracaoPage from '../views/InteracaoPage.vue';
import DesempenhoPage from '../views/DesempenhoPage.vue';
import BuscaAtivaPage from '@/views/BuscaAtivaPage.vue';
import ChatItsPage from '@/views/ChatItsPage.vue';
import AjudaRedu from '@/views/chat/AjudaRedu.vue';
import ChatItsRedu from '@/views/chat/ChatItsRedu.vue';
import TutoresHumanos from '@/views/chat/TutoresHumanos.vue';

import { RouteRecordRaw } from 'vue-router';

const routes: Array<RouteRecordRaw> = [
  {
    path: '/its/interacao',
    name: 'Interacao',
    component: InteracaoPage,
  },
  {
    path: '/its/desempenho',
    name: 'Desempenho',
    component: DesempenhoPage,
  },
  {
    path: '/its/busca',
    name: 'BuscaAtiva',
    component: BuscaAtivaPage,
  },
  {
    path: '/its/chat',
    name: 'Chat-Its',
    component: ChatItsPage,
  },
  {
    path: '/its/ajuda',
    name: 'Ajuda Redu',
    component: AjudaRedu,
  },
  {
    path: '/its/chatits',
    name: 'Chat-Its.Redu',
    component: ChatItsRedu,
  },
  {
    path: '/its/tutores',
    name: 'TutoresHumanos',
    component: TutoresHumanos,
  },
  {
    path: '/its/:id',
    component: () => import ('../views/ItsReduPage.vue')
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

export default router
