# MCP Standards Server Web UI

A modern web interface for browsing and interacting with the MCP Standards Server, featuring real-time updates, advanced search, and interactive rule testing.

## Features

### 🔍 Standards Browser
- Hierarchical tree view of all standards categories
- Detailed standard viewer with syntax highlighting
- Version history and comparison
- Export functionality (Markdown, PDF)

### 🔎 Advanced Search
- Semantic search across all standards
- Filter by category, tags, and attributes
- Search result highlighting
- Save and manage search queries

### 🧪 Rule Testing
- Interactive project context builder
- Real-time standard recommendations
- Visual rule flow diagrams
- Implementation guidance

### 🚀 Real-time Updates
- WebSocket integration for live updates
- Server connection status monitoring
- Automatic reconnection handling

## Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **WebSockets**: Real-time bidirectional communication
- **Redis**: Caching and session management
- **SQLite/PostgreSQL**: Data persistence

### Frontend
- **React 18**: Modern UI library with TypeScript
- **Material-UI**: Professional component library
- **React Router**: Client-side routing
- **Axios**: HTTP client
- **React Syntax Highlighter**: Code highlighting

## Quick Start

### Development

1. **Install dependencies:**
```bash
# Backend
cd web/backend
pip install -r requirements.txt

# Frontend
cd web/frontend
npm install
```

2. **Start the backend:**
```bash
cd web/backend
uvicorn main:app --reload --port 8000
```

3. **Start the frontend:**
```bash
cd web/frontend
npm start
```

4. **Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Production

Use Docker Compose for production deployment:

```bash
cd web
docker-compose up -d
```

Access at http://localhost (port 80)

## Project Structure

```
web/
├── backend/              # FastAPI backend
│   ├── main.py          # Application entry point
│   ├── models.py        # Pydantic models
│   ├── auth.py          # Authentication logic
│   └── requirements.txt # Python dependencies
│
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # Reusable components
│   │   ├── pages/       # Page components
│   │   ├── contexts/    # React contexts
│   │   ├── services/    # API services
│   │   ├── hooks/       # Custom hooks
│   │   └── types/       # TypeScript types
│   ├── public/          # Static assets
│   └── package.json     # Node dependencies
│
└── deployment/          # Deployment configuration
    ├── Dockerfile.*     # Container definitions
    ├── nginx.conf       # Nginx configuration
    └── DEPLOYMENT.md    # Deployment guide
```

## API Endpoints

### Standards
- `GET /api/standards` - Get all standards
- `GET /api/standards/{id}` - Get standard by ID
- `GET /api/categories` - Get all categories
- `GET /api/tags` - Get all tags

### Search
- `POST /api/search` - Search standards
- `POST /api/analyze` - Analyze project context

### Export
- `GET /api/export/{id}` - Export standard

### WebSocket
- `WS /ws` - Real-time updates

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Backend
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///./app.db
REDIS_URL=redis://localhost:6379

# Frontend
REACT_APP_API_URL=http://localhost:8000
```

## Development

### Code Style
- Backend: Black, isort, flake8
- Frontend: ESLint, Prettier

### Testing
```bash
# Backend
pytest

# Frontend
npm test
```

### Building for Production
```bash
# Frontend
npm run build

# Docker
docker-compose build
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- GitHub Issues: Report bugs and request features
- Documentation: See `/docs` for detailed guides
- Community: Join our Discord server