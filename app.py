

import chess
import chess.svg
import streamlit as st
import google.generativeai as genai
import time

# Initialize session state variables
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = None
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "made_move" not in st.session_state:
    st.session_state.made_move = False
if "move_history" not in st.session_state:
    st.session_state.move_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "max_turns" not in st.session_state:
    st.session_state.max_turns = 5
if "current_board_svg" not in st.session_state:
    st.session_state.current_board_svg = None
if "game_in_progress" not in st.session_state:
    st.session_state.game_in_progress = False
if "move_counter" not in st.session_state:
    st.session_state.move_counter = 0
if "game_mode" not in st.session_state:
    st.session_state.game_mode = "auto"
if "user_move" not in st.session_state:
    st.session_state.user_move = ""
if "invalid_move_message" not in st.session_state:
    st.session_state.invalid_move_message = None

# Sidebar for configuration
st.sidebar.title("Chess Agent Configuration")
gemini_api_key = st.sidebar.text_input("Enter your Gemini API key:", type="password")
if gemini_api_key:
    st.session_state.gemini_api_key = gemini_api_key
    try:
        genai.configure(api_key=gemini_api_key)
        st.sidebar.success("Gemini API key saved and configured!")
    except Exception as e:
        st.sidebar.error(f"Error configuring Gemini API: {e}")

# Model selection
model_options = ["gemini-1.5-flash", "gemini-2.0-flash"]
selected_model = st.sidebar.selectbox(
    "Select Gemini model:",
    options=model_options,
    index=0
)

# Game mode selection
game_mode_options = ["auto", "manual"]
selected_game_mode = st.sidebar.selectbox(
    "Select game mode:",
    options=game_mode_options,
    index=0,
    help="Auto: Two AI agents play against each other. Manual: You play against the AI."
)
st.session_state.game_mode = selected_game_mode

if selected_game_mode == "manual":
    player_color_options = ["white", "black"]
    player_color = st.sidebar.selectbox(
        "Choose your color:",
        options=player_color_options,
        index=0
    )
    st.session_state.player_color = player_color

st.sidebar.info("""
For a complete chess game with potential checkmate, it would take max_turns > 200 approximately.
However, this will consume significant API credits and a lot of time.
For demo purposes, using 5-10 turns is recommended.
""")

max_turns_input = st.sidebar.number_input(
    "Enter the number of turns (max_turns):",
    min_value=1,
    max_value=1000,
    value=st.session_state.max_turns,
    step=1
)

if max_turns_input:
    st.session_state.max_turns = max_turns_input
    st.sidebar.success(f"Max turns of total chess moves set to {st.session_state.max_turns}!")

st.title("Chess with Gemini AI Agents")

# Chess game functions
def get_board_state():
    """Return the current board state as a string representation."""
    return str(st.session_state.board)

def available_moves() -> str:
    """Get all legal moves in the current position."""
    available_moves = [str(move) for move in st.session_state.board.legal_moves]
    return "Available moves are: " + ", ".join(available_moves)

def execute_move(move: str) -> str:
    """Execute a chess move and update the board state."""
    try:
        chess_move = chess.Move.from_uci(move)
        if chess_move not in st.session_state.board.legal_moves:
            return f"Invalid move: {move}. Please check available moves."
        
        # Update board state
        st.session_state.board.push(chess_move)
        st.session_state.made_move = True
        st.session_state.move_counter += 1

        # Generate and store board visualization
        board_svg = chess.svg.board(
            st.session_state.board,
            arrows=[(chess_move.from_square, chess_move.to_square)],
            fill={chess_move.from_square: "gray"},
            size=400
        )
        st.session_state.current_board_svg = board_svg
        st.session_state.move_history.append((board_svg, st.session_state.move_counter))

        # Get piece information
        moved_piece = st.session_state.board.piece_at(chess_move.to_square)
        piece_unicode = moved_piece.unicode_symbol() if moved_piece else "?"
        piece_type_name = chess.piece_name(moved_piece.piece_type) if moved_piece else "unknown"
        piece_name = piece_type_name.capitalize() if moved_piece and piece_unicode.isupper() else piece_type_name
        
        # Generate move description
        from_square = chess.SQUARE_NAMES[chess_move.from_square]
        to_square = chess.SQUARE_NAMES[chess_move.to_square]
        move_desc = f"Moved {piece_name} ({piece_unicode}) from {from_square} to {to_square}."
        
        if st.session_state.board.is_checkmate():
            winner = 'White' if st.session_state.board.turn == chess.BLACK else 'Black'
            move_desc += f"\nCheckmate! {winner} wins!"
        elif st.session_state.board.is_stalemate():
            move_desc += "\nGame ended in stalemate!"
        elif st.session_state.board.is_insufficient_material():
            move_desc += "\nGame ended - insufficient material to checkmate!"
        elif st.session_state.board.is_check():
            move_desc += "\nCheck!"

        return move_desc
    except ValueError:
        return f"Invalid move format: {move}. Please use UCI format (e.g., 'e2e4')."

def validate_move(move: str) -> bool:
    """Validate if a move is in UCI format and legal."""
    try:
        chess_move = chess.Move.from_uci(move)
        if chess_move in st.session_state.board.legal_moves:
            return True
        return False
    except ValueError:
        return False

# Gemini agent functions
def create_gemini_model():
    """Create and return a configured Gemini model."""
    if not st.session_state.gemini_api_key:
        return None
    
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        model = genai.GenerativeModel(
            model_name=selected_model,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 0,
                "max_output_tokens": 8192,
            }
        )
        return model
    except Exception as e:
        st.error(f"Error creating Gemini model: {e}")
        return None

def get_agent_move(agent_name, color):
    """Get a move from the Gemini AI agent."""
    model = create_gemini_model()
    if not model:
        return "Error: Gemini model not available"
    
    # Create the context for the agent
    board_state = get_board_state()
    legal_moves = available_moves()
    turn_text = "white" if st.session_state.board.turn == chess.WHITE else "black"
    
    # Add previous moves to the context
    context = (
        f"You are {agent_name}, a professional chess player playing as {color}. "
        f"The current chess board position is:\n{board_state}\n\n"
        f"It is {turn_text}'s turn to move.\n\n"
        f"{legal_moves}\n\n"
    )
    
    if len(st.session_state.chat_history) > 0:
        context += "Previous moves in this game:\n"
        for i, entry in enumerate(st.session_state.chat_history[-10:]):  # limit to last 10 moves to save context space
            context += f"{entry}\n"
    
    # Add instructions
    prompt = (
        f"{context}\n"
        f"Analyze the position and make a strong chess move. "
        f"You must respond with a valid UCI format move (e.g., 'e2e4') from the available moves. "
        f"First, briefly analyze the position (max 2 sentences). "
        f"Then provide your move in the format: MOVE: e2e4 (replacing e2e4 with your chosen move)"
    )
    
    try:
        # Request move from Gemini model
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Extract the move from the response
        if "MOVE:" in response_text:
            move_part = response_text.split("MOVE:")[1].strip()
            move = move_part.split()[0].strip()
            # Store the response for display
            st.session_state.chat_history.append(f"{agent_name} ({color}): {response_text}")
            return move
        else:
            # If format is not followed, try to find a valid move in the text
            for word in response_text.split():
                clean_word = word.strip(",.;:()")
                try:
                    move = chess.Move.from_uci(clean_word)
                    if move in st.session_state.board.legal_moves:
                        st.session_state.chat_history.append(f"{agent_name} ({color}): {response_text}")
                        return clean_word
                except ValueError:
                    continue
            
            # No valid move found, request again with stricter instructions
            retry_prompt = (
                f"{prompt}\n\n"
                f"Your previous response did not contain a valid move. "
                f"You MUST include a valid move from {legal_moves} in the format 'MOVE: e2e4'"
            )
            response = model.generate_content(retry_prompt)
            response_text = response.text
            
            if "MOVE:" in response_text:
                move_part = response_text.split("MOVE:")[1].strip()
                move = move_part.split()[0].strip()
                st.session_state.chat_history.append(f"{agent_name} ({color}): {response_text}")
                return move
            else:
                # If still no valid move, select a random legal move
                import random
                legal_moves_list = list(st.session_state.board.legal_moves)
                random_move = str(random.choice(legal_moves_list))
                st.session_state.chat_history.append(
                    f"{agent_name} ({color}): Failed to provide a valid move format. " 
                    f"Selecting random legal move: {random_move}"
                )
                return random_move
    except Exception as e:
        st.error(f"Error getting move from Gemini: {e}")
        import random
        legal_moves_list = list(st.session_state.board.legal_moves)
        random_move = str(random.choice(legal_moves_list))
        return random_move

# Function to make a single move in the game (Auto Mode)
def make_single_move():
    if not st.session_state.game_in_progress:
        return False
    
    if st.session_state.move_counter >= st.session_state.max_turns * 2:  # Max turns for both players
        st.session_state.game_in_progress = False
        return False
    
    # Determine current player
    current_color = "White" if st.session_state.board.turn == chess.WHITE else "Black"
    agent_name = f"Agent_{current_color}"
    
    # Get move from current agent
    move = get_agent_move(agent_name, current_color.lower())
    
    # Execute the move
    result = execute_move(move)
    st.session_state.chat_history.append(f"Game Master: {result}")
    
    # Check if game is over
    if (st.session_state.board.is_checkmate() or 
        st.session_state.board.is_stalemate() or 
        st.session_state.board.is_insufficient_material()):
        st.session_state.game_in_progress = False
        return False
    
    return True

# Function to make AI move in manual mode
def make_ai_move():
    if not st.session_state.game_in_progress:
        return False
    
    # Determine AI color
    ai_color = "black" if st.session_state.player_color == "white" else "white"
    agent_name = f"Agent_{ai_color.capitalize()}"
    
    # Get move from AI agent
    move = get_agent_move(agent_name, ai_color)
    
    # Execute the move
    result = execute_move(move)
    st.session_state.chat_history.append(f"Game Master: {result}")
    
    # Check if game is over
    if (st.session_state.board.is_checkmate() or 
        st.session_state.board.is_stalemate() or 
        st.session_state.board.is_insufficient_material()):
        st.session_state.game_in_progress = False
        return False
    
    return True

# Function to handle user's turn in manual mode
def user_turn():
    """Process the user's move in manual mode."""
    if not st.session_state.game_in_progress:
        return
    
    current_turn_color = "white" if st.session_state.board.turn == chess.WHITE else "black"
    
    # Check if it's user's turn
    if current_turn_color != st.session_state.player_color:
        # Make AI move if it's not user's turn
        make_ai_move()
        st.rerun()
        return
    
    # Display info about available moves
    st.info(f"It's your turn ({st.session_state.player_color}).")
    legal_moves = [str(move) for move in st.session_state.board.legal_moves]
    
    # User input for move
    user_move = st.text_input(
        "Enter your move in UCI format (e.g., e2e4):",
        key="user_move_input",
        help="Enter a move from one square to another, like e2e4 (moving from e2 to e4)"
    )
    
    # Display available legal moves
    with st.expander("Show legal moves", expanded=False):
        st.write(", ".join(legal_moves))
    
    # Process user's move
    if st.button("Make Move"):
        if not user_move:
            st.session_state.invalid_move_message = "Please enter a move."
            return
        
        if validate_move(user_move):
            # Execute the valid move
            result = execute_move(user_move)
            st.session_state.chat_history.append(f"Player ({st.session_state.player_color}): Made move {user_move}")
            st.session_state.chat_history.append(f"Game Master: {result}")
            
            # Check if game is over
            if (st.session_state.board.is_checkmate() or 
                st.session_state.board.is_stalemate() or 
                st.session_state.board.is_insufficient_material()):
                st.session_state.game_in_progress = False
            else:
                # AI makes its move immediately after user's valid move
                make_ai_move()
            
            st.rerun()
        else:
            st.session_state.invalid_move_message = f"Invalid move: {user_move}. Please enter a valid move in UCI format."

# Streamlit UI
if st.session_state.gemini_api_key:
    try:
        # Display game info based on mode
        if st.session_state.game_mode == "auto":
            st.info("""
            This chess game is played between two Gemini AI agents:
            - **Agent White**: A Gemini-powered chess player controlling white pieces
            - **Agent Black**: A Gemini-powered chess player controlling black pieces

            The game is managed by a **Game Master** that:
            - Validates all moves
            - Updates the chess board
            - Manages turn-taking between players
            - Provides legal move information
            """)
        else:  # manual mode
            st.info(f"""
            You are playing as **{st.session_state.player_color.capitalize()}** against a Gemini AI agent.
            - **You**: Control the {st.session_state.player_color} pieces
            - **Agent_{("black" if st.session_state.player_color == "white" else "white").capitalize()}**: 
              A Gemini-powered chess player controlling {("black" if st.session_state.player_color == "white" else "white")} pieces

            The game is managed by a **Game Master** that:
            - Validates all moves
            - Updates the chess board
            - Manages turn-taking between players
            """)

        # Current board display container
        current_board_container = st.container()
        with current_board_container:
            st.subheader("Current Board")
            if st.session_state.current_board_svg:
                st.image(st.session_state.current_board_svg)
            else:
                initial_board_svg = chess.svg.board(st.session_state.board, size=400)
                st.image(initial_board_svg)

        # Display invalid move message if there is one
        if st.session_state.invalid_move_message:
            st.error(st.session_state.invalid_move_message)
            st.session_state.invalid_move_message = None  # Clear the message after showing it once

        # Game controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Game"):
                # Reset game state
                st.session_state.board.reset()
                st.session_state.made_move = False
                st.session_state.move_history = []
                st.session_state.current_board_svg = chess.svg.board(st.session_state.board, size=400)
                st.session_state.chat_history = []
                st.session_state.game_in_progress = True
                st.session_state.move_counter = 0
                
                if st.session_state.game_mode == "auto":
                    st.info("Game started! The Gemini AI agents will now play against each other.")
                else:
                    st.info(f"Game started! You are playing as {st.session_state.player_color}.")
                
                # If in manual mode and player is black, make the first move with AI
                if st.session_state.game_mode == "manual" and st.session_state.player_color == "black":
                    make_ai_move()
                
                st.rerun()
        
        with col2:
            if st.button("Reset Game"):
                st.session_state.board.reset()
                st.session_state.made_move = False
                st.session_state.move_history = []
                st.session_state.current_board_svg = chess.svg.board(st.session_state.board, size=400)
                st.session_state.chat_history = []
                st.session_state.game_in_progress = False
                st.session_state.move_counter = 0
                st.success("Game reset! Click 'Start Game' to begin a new game.")
                st.rerun()

        # Game UI based on mode
        if st.session_state.game_in_progress:
            if st.session_state.game_mode == "auto":
                # Auto mode controls
                if st.button("Make Next Move"):
                    continue_game = make_single_move()
                    st.rerun()
                
                # Auto-play functionality
                auto_play = st.checkbox("Auto-play (will continue until game ends)")
                if auto_play:
                    with st.spinner("Game in progress..."):
                        while st.session_state.game_in_progress:
                            continue_game = make_single_move()
                            st.rerun()
            else:
                # Manual mode - handle user's turn
                user_turn()
        
        # Agent conversation display
        with st.expander("Game Log", expanded=True):
            for message in st.session_state.chat_history:
                st.text(message)
        
        # Move history display
        with st.expander("Move History", expanded=True):
            for svg, move_num in st.session_state.move_history:
                if st.session_state.game_mode == "auto":
                    move_by = "Agent White" if move_num % 2 == 1 else "Agent Black"
                else:  # manual mode
                    if st.session_state.player_color == "white":
                        move_by = "Player" if move_num % 2 == 1 else "Agent Black"
                    else:
                        move_by = "Agent White" if move_num % 2 == 1 else "Player"
                
                st.write(f"Move {(move_num + 1) // 2} by {move_by}")
                st.image(svg)

    except Exception as e:
        st.error(f"An error occurred: {e}. Please check your API key and try again.")

else:
    st.warning("Please enter your Gemini API key in the sidebar to start the game.")