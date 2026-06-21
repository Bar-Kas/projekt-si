import random
from src.agent.poker_agent import NNAgent
from src.training.metrics import profile_agent_strategy, print_strategy_report

def run_temporary_demo():
    print("Inicjalizacja demonstracji modułu analizy...")
    
    # 1. Tworzymy losowy genom (chromosom) o długości 210 tak jak zrobi to algorytm genetyczny
    losowy_chromosom = [random.uniform(-1.0, 1.0) for _ in range(210)]
    
    # 2. Powołujemy do życia agenta z tymi wagami
    agent = NNAgent(losowy_chromosom)
    
    # 3. Pobieramy jego profil strategiczny przy użyciu naszego modułu metryk
    profil = profile_agent_strategy(agent)
    
    # 4. Drukowanie raportu
    print_strategy_report(profil)

if __name__ == "__main__":
    run_temporary_demo()