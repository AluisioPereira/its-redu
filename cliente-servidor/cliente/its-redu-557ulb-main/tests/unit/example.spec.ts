import { mount } from '@vue/test-utils'
import ItsReduPage from '@/views/ItsReduPage.vue'
import { describe, expect, test } from 'vitest'

describe('ItsReduPage.vue', () => {
  test('renders its view', () => {
    const mockRoute = {
      params: {
        id: 'Desempenho'
      }
    }
    const wrapper = mount(ItsReduPage, {
      global: {
        mocks: {
          $route: mockRoute
        }
      }
    })
    expect(wrapper.text()).toMatch('Explore UI Components')
  })
})
