# SportSphere AI Agent

**A Conversational AI Assistant for Sports Enthusiasts**  
*Get real-time sports information through natural voice or text interactions*

## 🏆 Project Overview
SportSphere is a multimodal AI agent that provides comprehensive information about:
- ⚾ **MLB** (Baseball) - *Currently Active*
- 🏀 **NBA** (Basketball) - *Coming Soon*
- ⚽ **Global Soccer** (Multiple leagues/countries) - *Coming Soon*

## 🌟 Core Features
- **Multi-Sport Intelligence**: Query players, teams, schedules, and stats across 3 major sports
- **Voice-First Interaction**:
  - 🎤 Speech recognition for natural language input
  - 🔊 Text-to-speech (TTS) for audible responses
- **Real-Time Data Integration**:  
  - Live scores
  - Team rosters
  - Player statistics
  - League standings
- **Multi-Language Support** (Phase 2):
  - English (primary)
  - Spanish (Q2 2024)
  - French (Q3 2024)

## 🚀 Current Implementation (MLB Focus)
```python
MLB Features              Status
───────────────────────────────────────
Team Information          ✅ Live
Player Rosters            ✅ Live
Game Schedules            🚧 In Development
Live Score Updates        🚧 In Development
Historical Data           🚧 In Development
```

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation
git clone https://github.com/obinopaul/sportsphere-ai.git
cd sportsphere-ai
pip install -r requirements.txt

### Configuration
Create .env file:

```ini
OPENAI_API_KEY=your_openai_key
SPORTSDATA_API_KEY=your_sportsdata_key
```
