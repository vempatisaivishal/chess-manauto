# Chess with Gemini AI Agents


A powerful and interactive chess application that enables both AI vs AI gameplay and human vs AI gameplay. This application leverages the Gemini AI to create intelligent chess agents that can analyze positions and make strategic moves.

## Features

- **Dual Game Modes**:
  - **Auto Mode**: Watch two Gemini AI agents play against each other
  - **Manual Mode**: Play against a Gemini AI agent as either white or black

- **Interactive Chess Board**: 
  - Real-time visual representation of the chess board
  - Move highlighting and arrows showing the last move played
  - SVG-based rendering for crisp visuals at any size

- **Intelligent AI Gameplay**:
  - Powered by Google's Gemini AI models (gemini-1.5-flash or gemini-2.0-flash)
  - Agents analyze the position before making moves
  - AI agents provide brief analysis of their thinking process

- **Complete Game Tracking**:
  - Detailed game log showing agent reasoning
  - Visual move history with board positions after each move
  - Proper game state tracking (check, checkmate, stalemate, etc.)

- **Customization Options**:
  - Select your preferred Gemini model
  - Choose your playing color in manual mode
  - Control the maximum number of turns
  - Auto-play functionality for hands-free AI vs AI gameplay

## Getting Started

### Prerequisites

- A Gemini API key (from Google AI Studio)
- Python 3.7 or higher
- Required Python packages (installable via pip):
  - streamlit
  - chess
  - google-generativeai

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/vempatisaivishal/chess-manauto.git
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Usage

1. Enter your Gemini API key in the sidebar
2. Select your preferred Gemini model
3. Choose a game mode (auto or manual)
4. Set the maximum number of turns
5. Click "Start Game" to begin playing
6. In manual mode, enter moves in UCI format (e.g., "e2e4")
7. In auto mode, use "Make Next Move" or toggle "Auto-play"

## Technical Details

- Built with Streamlit for the interactive web interface
- Uses the python-chess library for game state management
- Integrates with Google's Generative AI models through the google-generativeai package
- Implements chess move validation and execution with proper UCI format
- Provides SVG-based visualization of the chess board with move highlighting

## Project Structure

```
chess-gemini/
│
├── app.py          # Main Streamlit application 
├── requirements.txt # Dependencies list
├── README.md       # Project documentation
```


## Acknowledgements

- [python-chess](https://python-chess.readthedocs.io/) for the chess game logic
- [Streamlit](https://streamlit.io/) for the web application framework
- [Google Generative AI](https://ai.google.dev/) for the Gemini models

## Live Demo

Try it out now at: [https://chess-auto-manual.streamlit.app/](https://chess-auto-manual.streamlit.app/)
