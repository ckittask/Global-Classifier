const { render, screen } = require('@testing-library/react');
const Header = require('../Header');

test('renders header component', () => {
	render(<Header />);
	const linkElement = screen.getByText(/header/i);
	expect(linkElement).toBeInTheDocument();
});