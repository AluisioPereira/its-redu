describe('My First Test', () => {
  it('Visits the app root url', () => {
    cy.visit('/its/Interacao')
    cy.contains('#container', 'Interacao')
  })
})
