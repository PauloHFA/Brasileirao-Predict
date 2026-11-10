from src.entity_resolution.teams import TeamResolver


def test_resolve_standard_aliases():
    resolver = TeamResolver()
    
    # Flamengo variants
    fla1 = resolver.resolve("Flamengo")
    fla2 = resolver.resolve("C.R. Flamengo")
    fla3 = resolver.resolve("Mengo")
    fla4 = resolver.resolve("FLA")
    assert fla1 is not None and fla2 is not None and fla3 is not None and fla4 is not None
    assert fla1["id"] == fla2["id"] == fla3["id"] == fla4["id"] == 1

    # Palmeiras variants
    pal1 = resolver.resolve("Palmeiras")
    pal2 = resolver.resolve("S.E. Palmeiras")
    pal3 = resolver.resolve("Verdão")
    assert pal1 is not None and pal2 is not None and pal3 is not None
    assert pal1["id"] == pal2["id"] == pal3["id"] == 2

def test_do_not_collapse_distinct_clubs():
    resolver = TeamResolver()

    # Botafogo RJ vs SP
    bota_rj = resolver.resolve("Botafogo")
    bota_sp = resolver.resolve("Botafogo-SP")
    assert bota_rj is not None
    assert bota_sp is not None
    assert bota_rj["id"] != bota_sp["id"]
    assert bota_rj["slug"] == "botafogo_rj"
    assert bota_sp["slug"] == "botafogo_sp"

    # América MG vs RN
    ame_mg = resolver.resolve("América Mineiro")
    ame_rn = resolver.resolve("América de Natal")
    assert ame_mg is not None
    assert ame_rn is not None
    assert ame_mg["id"] != ame_rn["id"]
    assert ame_mg["slug"] == "america_mg"
    assert ame_rn["slug"] == "america_rn"

    # Atlético MG vs GO vs Athletico PR
    cam = resolver.resolve("Atlético Mineiro")
    cap = resolver.resolve("Athletico Paranaense")
    dragao = resolver.resolve("Atlético Goianiense")
    assert cam is not None and cap is not None and dragao is not None
    assert cam["id"] != cap["id"]
    assert cam["id"] != dragao["id"]
    assert cap["id"] != dragao["id"]
