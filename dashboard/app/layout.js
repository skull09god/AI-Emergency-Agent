import "./globals.css";

export const metadata = {
  title: "AI Emergency Dashboard",
  description: "Dashboard for emergency detection system",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="flex">
        {/* Sidebar */}
        <div className="w-64 h-screen bg-gray-900 text-white p-6 space-y-6">
          <h1 className="text-2xl font-bold">Dashboard</h1>

          <nav className="flex flex-col space-y-4">
            <a href="/" className="hover:text-gray-300">Home</a>
            <a href="/alerts" className="hover:text-gray-300">Alerts</a>
          </nav>
        </div>

        {/* Main Page Content */}
        <main className="flex-1 bg-gray-100 p-8">{children}</main>
      </body>
    </html>
  );
}
