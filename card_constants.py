# Definisi template untuk kartu
CARD_RANKS = [
    'ace', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    'jack', 'queen', 'king'
]

CARD_SUITS = [
    'hearts', 'diamonds', 'clubs', 'spades'
]

# Generate semua kombinasi kartu yang valid
VALID_CARDS = [
    f"{rank}_of_{suit}" for rank in CARD_RANKS
    for suit in CARD_SUITS
]

# Dictionary untuk validasi input
CARD_LABELS = {card: True for card in VALID_CARDS}


def get_valid_label(input_label):
    """
    Memvalidasi dan memformat label kartu
    """
    formatted_label = input_label.lower().replace(' ', '_')
    if formatted_label in CARD_LABELS:
        return formatted_label
    return None


def print_valid_labels():
    """
    Menampilkan semua label kartu yang valid
    """
    print("\nValid card labels:")
    for suit in CARD_SUITS:
        print(f"\n{suit.upper()}:")
        cards = [card for card in VALID_CARDS if suit in card]
        print(", ".join(cards))
