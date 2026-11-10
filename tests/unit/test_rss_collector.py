from src.ingestion.rss_collector import RSSNewsCollector, clean_html_tags

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Notícias Esportivas</title>
    <link>http://exemplo.com</link>
    <item>
      <title>Flamengo confirma treino com titulares antes do clássico</title>
      <link>https://exemplo.com/noticia-1</link>
      <description><![CDATA[<p>O elenco do Flamengo realizou atividade técnica no Ninho do Urubu.</p>]]></description>
      <pubDate>Wed, 10 Apr 2024 15:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Palmeiras viaja para enfrentar adversário fora de casa</title>
      <link>https://exemplo.com/noticia-2</link>
      <description>Equipe alviverde finalizou os preparativos na Academia de Futebol.</description>
      <pubDate>Wed, 10 Apr 2024 16:30:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

def test_clean_html_tags():
    raw = "<p>Texto <b>com tags</b> e <a href='link'>links</a>.</p>"
    clean = clean_html_tags(raw)
    assert clean == "Texto com tags e links."

def test_match_teams_in_text():
    collector = RSSNewsCollector()
    teams = collector.match_teams_in_text("Flamengo e Palmeiras disputam a liderança da Série A.")
    slugs = [t["slug"] for t in teams]
    assert "flamengo" in slugs
    assert "palmeiras" in slugs

def test_do_not_falsely_match_unmentioned_teams():
    collector = RSSNewsCollector()
    teams = collector.match_teams_in_text("Treino tático sem novidades no CT.")
    assert len(teams) == 0
