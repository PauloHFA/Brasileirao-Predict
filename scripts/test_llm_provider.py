"""
Script utilitário para testar e validar o provedor de LLM configurado.
Testa conectividade, latência e executa uma extração de homologação estruturada.
"""
import sys
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

from src.config import settings  # noqa: E402
from src.llm_extraction.extractor import get_extractor  # noqa: E402
from src.llm_extraction.providers import check_llm_provider_health  # noqa: E402


def main():
    print("=" * 60)
    print("   🤖 TESTE E DIAGNÓSTICO DO PROVEDOR DE LLM")
    print("=" * 60)

    print(f"Provedor Configurado: {settings.LLM_PROVIDER}")
    print(f"Modelo Configurado:   {settings.LLM_MODEL}")
    print(f"Base URL (se Ollama): {settings.LLM_BASE_URL}")

    print("\n1. Checando Conectividade do Provedor...")
    health = check_llm_provider_health()
    print(f" -> Status: {health.get('status')}")
    print(f" -> Mensagem: {health.get('message')}")
    if "latency_ms" in health:
        print(f" -> Latência: {health['latency_ms']} ms")
    if "available_models" in health:
        print(f" -> Modelos Encontrados no Ollama: {health['available_models']}")

    print("\n2. Executando Extração de Homologação com Rubrica...")
    extractor = get_extractor()
    sample_text = (
        "O departamento médico do Flamengo confirmou na manhã desta quinta-feira "
        "que o centroavante Pedro sofreu uma lesão ligamentar no joelho e está fora "
        "da partida do fim de semana pelo Brasileirão."
    )
    result = extractor.extract(
        doc_id="DOC_TEST_01",
        title="Pedro sofre lesão no joelho e desfalca o Flamengo",
        text_content=sample_text,
        team_slug="flamengo",
        team_name="Flamengo",
        as_of_at="2024-04-10T12:00:00Z"
    )

    print(" -> Schema Version:", result.schema_version)
    print(" -> Features Extraídas:")
    print(f"    • pressao_crise:        {result.features.pressao_crise}")
    print(f"    • impacto_desfalques:   {result.features.impacto_desfalques}")
    print(f"    • prioridade_rotacao:   {result.features.prioridade_rotacao}")
    print(" -> Evidências Citadas:")
    for ev in result.evidence:
        print(f"    • [{ev.id}] ({ev.certainty}): \"{ev.quote}\"")

    print("\n" + "=" * 60)
    print("   ✅ DIAGNÓSTICO DO PROVEDOR CONCLUÍDO COM SUCESSO")
    print("=" * 60)

if __name__ == "__main__":
    main()
