from enum import Enum

from typing import Optional, List, TypeVar, Generic

from pydantic import BaseModel, Field

T = TypeVar("T")


class LabelWithReferences(BaseModel, Generic[T]):
    """
    Erfasst pathologische Befunde mit verpflichtenden Referenzen aus dem Befundtext.

    Für LLM-Extraktion: Diese Klasse wird für definitive pathologische Befunde verwendet,
    die immer eine Textquelle im Befund haben müssen. LLMs sollen hier sowohl den
    pathologischen Befund als auch die entsprechenden Textstellen extrahieren.

    Beispiele aus Sprunggelenk-MRT Befunden:
    - "umschriebener Knorpeldefekt medial am Talus" → items: VOLLSCHICHTDEFEKT, references: ["umschriebener Knorpeldefekt medial am Talus"]
    - "Ruptur des vorderen Talofibularbandes" → items: KOMPLETTRUPTUR, references: ["Ruptur des vorderen Talofibularbandes"]
    - "mäßiger Gelenkerguss" → items: MODERAT, references: ["mäßiger Gelenkerguss"]
    """

    value: T = Field(
        description="Der pathologische Befund aus der vordefinierten Enumeration"
    )
    references: list[str] = Field(
        description="Wörtliche Textstellen aus dem MRT-Befund, die diesen Befund belegen"
    )


class LabelWithOptionalReferences(BaseModel, Generic[T]):
    """
    Erfasst pathologische Befunde mit optionalen Referenzen - besonders für Negativbefunde.

    Für LLM-Extraktion: Diese Klasse wird verwendet wenn:
    1. NEGATIVBEFUNDE vorliegen (z.B. "KEIN Gelenkerguss") - hier sind KEINE Referenzen erforderlich
    2. Befunde vorliegen, die möglicherweise nicht explizit im Text erwähnt werden
    3. Bewertungen/Graduierungen, die aus dem Kontext abgeleitet werden können

    WICHTIG für Negativbefunde: Bei Werten wie GelenkergussGrad.KEIN, SynovitisGrad.KEIN, etc.
    sind references = None oder [] vollkommen ausreichend, da diese Befunde oft durch
    ABWESENHEIT von pathologischen Beschreibungen charakterisiert sind.

    Beispiele aus Sprunggelenk-MRT Befunden:
    - Kein Gelenkerguss sichtbar → items: KEIN, references: None (Negativbefund)
    - "geringer Erguss" → items: GERING, references: ["geringer Erguss"]
    - Normale Synovialis ohne Erwähnung → items: KEIN, references: None (Negativbefund)
    - "Stabiles osteochondrales Fragment" → items: STABIL, references: ["Stabiles osteochondrales Fragment"]
    """

    value: T = Field(
        description="Der pathologische Befund oder Negativbefund aus der vordefinierten Enumeration"
    )
    references: Optional[list[str]] = Field(
        default=None,
        description="Optionale Textstellen aus dem MRT-Befund. Bei Negativbefunden (z.B. KEIN, NORMAL) können references leer bleiben.",
    )


# =========================

# ======== ENUMS ==========

# =========================


class GelenkergussGrad(Enum):
    """
    Graduierung von Gelenkergüssen im MRT.

    LLM-Extraktionshilfe: Suche nach Begriffen wie "Erguss", "Flüssigkeit", "Distension",
    "Kapselspannung" kombiniert mit Gradadjektiven.
    """

    KEIN = "Kein"  # Negativbefund: "kein Erguss", "keine Flüssigkeit", "reguläre Kapselverhältnisse"
    GERING = "Gering"  # "geringer/geringfügiger/minimaler Erguss", "wenig Flüssigkeit"
    MODERAT = "Moderat"  # "mäßiger/moderater Erguss", "deutlicher Erguss", "vermehrte Gelenkflüssigkeit"
    AUSGEPRAEGT = (
        "Ausgepraegt"  # "ausgeprägter/erheblicher/massiver Erguss", "starke Distension"
    )


class SynovitisGrad(Enum):
    """
    Graduierung von Synovitis (Synovialmembran-Entzündung) im MRT.

    LLM-Extraktionshilfe: Suche nach "Synovitis", "Synovialmembran", "Synovia",
    "entzündliche Veränderungen", "KM-Enhancement der Synovia".
    """

    KEIN = "Kein"  # Negativbefund: "keine Synovitis", "unauffällige Synovia", "regelrechte Synovialmembran"
    LEICHT = "Leicht"  # "leichte/geringgradige Synovitis", "diskrete synoviale Veränderungen"
    MODERAT = (
        "Moderat"  # "mäßige/moderate Synovitis", "deutliche synoviale Proliferation"
    )
    SCHWER = (
        "Schwer"  # "schwere/ausgeprägte Synovitis", "massive synoviale Hyperplasie"
    )


class TenosynovitisGrad(Enum):
    """
    Graduierung von Tenosynovitis (Sehnenscheiden-Entzündung) im MRT.

    LLM-Extraktionshilfe: Suche nach "Tenosynovitis", "Sehnenscheide", "peritendinöses Ödem",
    "Flüssigkeit um die Sehne", "Enhancement der Sehnenscheide".
    """

    KEIN = "Kein"  # Negativbefund: "keine Tenosynovitis", "unauffällige Sehnenscheiden", "ohne peritendinöses Ödem"
    LEICHT = "Leicht"  # "leichte/geringgradige Tenosynovitis", "diskrete Flüssigkeit in der Sehnenscheide"
    MODERAT = "Moderat"  # "mäßige/moderate Tenosynovitis", "deutliche Sehnenscheidenerweiterung"
    SCHWER = "Schwer"  # "schwere/ausgeprägte Tenosynovitis", "massive Sehnenscheidenverdickung"


class Sehnenpathologie(Enum):
    """
    Pathologische Veränderungen der Sehnen im MRT.

    LLM-Extraktionshilfe: Suche nach Sehnenamen (Achilles, Peroneus, Tibialis, etc.)
    kombiniert mit pathologischen Begriffen.
    """

    TENDINOPATHIE = "Tendinopathie"  # "Tendinopathie", "Tendinose", "degenerative Sehnenveränderung", "Sehnenverdickung"
    TENOSYNOVITIS = "Tenosynovitis"  # "Tenosynovitis", "Sehnenscheidenentzündung", "peritendinöses Ödem"
    PARTIALRUPTUR = "Partialruptur"  # "Partialruptur", "Teilruptur", "inkompletter Riss", "partielle Kontinuitätsunterbrechung"
    LAENGSRISS = "Laengsriss"  # "Längsriss", "longitudinaler Riss", "Spaltbildung", "intrastendinöse Läsion"
    KOMPLETTRUPTUR = "Komplettruptur"  # "Komplettruptur", "Totalruptur", "kompletter Riss", "vollständige Kontinuitätsunterbrechung"
    SUBLUXATION = "Subluxation"  # "Subluxation", "Teilverrenkung", "instabile Lage", "Dislokation"
    LUXATION = (
        "Luxation"  # "Luxation", "Verrenkung", "komplette Verlagerung", "Ausrenkung"
    )
    KALZIFIZIERTE_TENDINOSE = "Kalzifizierte Tendinose"  # "Kalzifikation", "Verkalkung", "kalzifizierte Tendinose", "Kalkeinlagerung"
    POSTOPERATIV = "Postoperative Veraenderung"  # "postoperativ", "Z.n. Operation", "Nahtmaterial", "operativ versorgt"
    NICHT_ABGRENZBAR = "Nicht abgrenzbar"  # "nicht abgrenzbar", "schwer beurteilbar", "unklare Signalveränderung"


class AchillesBefund(Enum):
    """
    Spezifische Achillessehnen-assoziierte Pathologien.

    LLM-Extraktionshilfe: Suche nach Achillessehnen-spezifischen Begriffen
    kombiniert mit anatomischen Strukturen um die Achillessehne.
    """

    HAGLUND_DEFORMITAET = "Haglund-Deformitaet"  # "Haglund-Deformität", "Haglund-Exostose", "Pumpenbuckel", "posterosuperior calcaneal prominence"
    BURSITIS_SUBACHILLEA = "Bursitis subachillea"  # "Bursitis subachillea", "retro-achillea Bursitis", "Schleimbeutelentzündung hinter Achillessehne"
    BURSITIS_SUBCUTANEA = "Bursitis subcutanea calcanea"  # "subcutane Bursitis", "Bursitis subcutanea calcanea", "oberflächliche Schleimbeutelentzündung"
    ENTHESIOPATHIE = "Enthesiopathie"  # "Enthesiopathie", "Ansatztendinopathie", "Achillessehnen-Ansatz-Entzündung", "insertionale Tendinopathie"
    PERITENDINITIS = "Peritendinitis"  # "Peritendinitis", "Paratendinitis", "Entzündung um die Achillessehne", "peritendinöse Entzündung"
    OEDEM_IM_KAGER_FETTKÖRPER = "Oedem im Kager Fettkörper"  # "Kager-Fettkörper-Ödem", "prä-achilleales Ödem", "Kager fat pad edema"


class Bandpathologie(Enum):
    """
    Pathologische Veränderungen der Bänder im MRT.

    LLM-Extraktionshilfe: Suche nach Bandnamen (LFTA, LFC, LFTP, Syndesmose, etc.)
    kombiniert mit pathologischen Begriffen.
    """

    VERDICKT = "Verdickt"  # "verdickt", "Verdickung", "knotig verdickt", "Auftreibung"
    OEDEMATOES = (
        "Oedematoes"  # "ödematös", "Ödem", "signalreich", "hyperintens", "entzündlich"
    )
    PARTIALRUPTUR = "Partialruptur"  # "Partialruptur", "Teilruptur", "inkompletter Riss", "partielle Läsion"
    KOMPLETTRUPTUR = "Komplettruptur"  # "Komplettruptur", "Totalruptur", "kompletter Riss", "durchtrennt"
    HEILUNGSZEICHEN = (
        "Heilungszeichen"  # "Heilungszeichen", "Reparatur", "Vernarbung", "regenerativ"
    )
    POSTOPERATIV_REKONSTRUKTION = "Z.n. Rekonstruktion"  # "Z.n. Rekonstruktion", "rekonstruiert", "Bandersatz", "postoperativ"
    MUCOIDE_DEGENERIERT = "mucoide degeneriert"  # "mucoide Degeneration", "schleimig degeneriert", "myxomatös"
    NICHT_ABGRENZBAR = "Nicht abgrenzbar"  # "nicht abgrenzbar", "schwer beurteilbar", "unklare Darstellung"


class Knorpelpathologie(Enum):
    """
    Knorpelpathologien im MRT - häufig im OSG an Talus und distaler Tibia.

    LLM-Extraktionshilfe: Suche nach "Knorpel", "chondral", "osteochondral"
    kombiniert mit Defekt-, Läsions- oder Degenerationsbegriffen.
    """

    TEILSCHICHTDEFEKT = "Fokaler teilschichtiger Defekt"  # "teilschichtiger Defekt", "oberflächlicher Defekt", "partieller Knorpeldefekt"
    VOLLSCHICHTDEFEKT = "Fokaler vollschichtiger Defekt"  # "vollschichtiger Defekt", "kompletter Knorpeldefekt", "bis zum Knochen reichend"
    OBERFLAECHENFIBRILLATION = "Oberflaechenfibrillation"  # "Oberflächenfibrillation", "aufgeraute Oberfläche", "fibrillär"
    FISSUR = "Fissur"  # "Fissur", "Riss", "Spalte", "Knorpelspalt"
    DELAMINATION = (
        "Delamination"  # "Delamination", "Ablösung", "Knorpelabhebung", "Abschälung"
    )
    CHONDROPATHIE_II_III = "Chondropathie Grad II/III"  # "Chondropathie Grad 2", "Grad 3", "mäßige Chondropathie"
    CHONDROPATHIE_IV = "Chondropathie Grad IV"  # "Chondropathie Grad 4", "schwere Chondropathie", "Knorpelglatze"
    AUSDUENNUNG = "Knorpelreduktion"  # "Knorpelreduktion", "Ausdünnung", "verdünnt", "verschmälert"
    OSTEOCHONDRALE_LAESION = "Osteochondrale Laesion"  # "osteochondrale Läsion", "Knorpel-Knochen-Defekt", "OCD"
    ZN_KNORPELERSATZ = "Z.n. Knorpelersatz"  # "Z.n. Knorpelersatz", "Knorpeltransplantation", "autologe Transplantation"
    ZN_MIKROFRAKTURIERUNG = "Z.n. Mikrofrakturierung"  # "Z.n. Mikrofrakturierung", "Pridie-Bohrung", "subchondrale Anbohrung"


class OsteochondraleStabilitaet(Enum):
    """
    Stabilität osteochondraler Läsionen - wichtig für Therapieentscheidung.

    LLM-Extraktionshilfe: Suche nach "stabil", "instabil", "Fragment", "Läsion"
    in Verbindung mit osteochondralen Befunden.
    """

    STABIL = "Stabil"  # "stabil", "fest verwachsen", "ohne Dislokation", "in situ"
    INSTABIL = "Instabil"  # "instabil", "beweglich", "disloziert", "loose body", "freies Fragment"
    NICHT_BEURTEILBAR = "Nicht beurteilbar"  # "nicht beurteilbar", "unklare Abgrenzung", "schwer evaluierbar"


class Knochenpathologie(Enum):
    """
    Pathologische Knochenveränderungen im MRT.

    LLM-Extraktionshilfe: Suche nach Knochennamen (Tibia, Fibula, Talus, Calcaneus, etc.)
    kombiniert mit pathologischen Signalveränderungen.
    """

    SUBCHONDRALES_KM_OEDEM = "Subchondrales Knochenmarkoedem"  # "subchondrales Ödem", "Knochenmarködem", "hyperintense Signalveränderung"
    MARKOEDEM = "Markraumoedem"  # "Markraumödem", "Knochenmarködem", "erhöhtes Signal in STIR", "bone bruise"
    SUBCHONDRALE_SKLEROSIERUNG = "Subchondrale Sklerosierung"  # "Sklerosierung", "Verdichtung", "hypointens", "verminderte Signalintensität"
    SUBCHONDRALE_ZYSTE = "Subchondrale Zyste"  # "subchondrale Zyste", "Geröllzyste", "zystische Läsion", "flüssigkeitsequivalent"
    FRAKTUR = (
        "Fraktur"  # "Fraktur", "Bruch", "Kontinuitätsunterbrechung", "Frakturlinie"
    )
    STRESSREAKTION = "Stressreaktion"  # "Stressreaktion", "Ermüdungsreaktion", "Stressfraktur", "Überlastung"
    NEKROSE = "Knochennekrose"  # "Knochennekrose", "Osteonekrose", "avaskuläre Nekrose", "ischämische Nekrose"
    OSTEOMYELITIS = "Osteomyelitis"  # "Osteomyelitis", "Knocheninfektion", "entzündliche Knochenveränderung"
    KONTUSION = "Knochenkontusion"  # "Knochenkontusion", "bone bruise", "Knochenmarködem nach Trauma"
    ANDERE = "Andere Knochenlaesion"  # "andere Läsion", "unspezifische Veränderung", "weitere Pathologie"


class _Rueckfussstellung(Enum):
    """
    Rückfußstellung - wichtig für biomechanische Bewertung.

    LLM-Extraktionshilfe: Suche nach "Rückfuß", "Alignment", "Achse", "Stellung"
    kombiniert mit Richtungsbegriffen.
    """

    NEUTRAL = "Neutral"  # Negativbefund: "neutrale Stellung", "gerade Achse", "physiologisches Alignment", "regelrechte Stellung"
    VALGUS = "Valgus"  # "Valgusstellung", "X-Bein-Stellung", "nach außen abgewinkelt", "laterale Deviation"
    VARUS = "Varus"  # "Varusstellung", "O-Bein-Stellung", "nach innen abgewinkelt", "mediale Deviation"


class CoalitioTyp(Enum):
    """
    Coalitio (knöcherne oder fibröse Fusion zwischen Fußwurzelknochen).

    LLM-Extraktionshilfe: Suche nach "Coalitio", "Fusion", "Synostose", "Verwachsung"
    zwischen spezifischen Fußwurzelknochen.
    """

    KEINE = "Keine"  # Negativbefund: "keine Coalitio", "keine Verwachsung", "getrennte Gelenkflächen", "regelrechte Gelenkräume"
    KALKANEO_NAVIKULAER = "Calcaneonaviculare Coalitio"  # "calcaneonavikuläre Coalitio", "Verwachsung Calcaneus-Naviculare"
    TALOKALKANEAL = "Talokalkaneare Coalitio"  # "talokalkaneare Coalitio", "Verwachsung Talus-Calcaneus", "subtalare Coalitio"
    TALONAVIKULAER = "Talonavikulare Coalitio"  # "talonavikuläre Coalitio", "Verwachsung Talus-Naviculare"
    ANDERE = "Andere"  # "andere Coalitio", "seltene Verwachsung", "weitere Synostose"


class AkzessorischesOssikel(Enum):
    """
    Akzessorische Ossikeln (zusätzliche Knochenkerne) im Sprunggelenkbereich.

    LLM-Extraktionshilfe: Suche nach "Os", "Ossikel", "akzessorisch",
    "zusätzlicher Knochenkern", spezifische Ossikelnamen.
    """

    OS_TRIGONUM = "Os trigonum"  # "Os trigonum", "Stieda-Prozess", "akzessorisches Ossikel posterior Talus"
    STIEDA_FORTSATZ = "Stieda-Fortsatz"  # "Stieda-Fortsatz", "Processus posterior tali", "verlängerter Talushals"
    OS_PERONEUM = "Os peroneum"  # "Os peroneum", "Ossikel in der Peroneus longus Sehne", "Sesamoid peroneal"
    OS_NAVICULARE = "Os naviculare accessorium"  # "Os naviculare accessorium", "akzessorisches Kahnbein", "zusätzliches Naviculare"
    OS_SUBTIBIALE = "Os subtibiale"  # "Os subtibiale", "Ossikel medial Tibia", "zusätzlicher Knochenkern medial"
    OS_SUBFIBULARE = "Os subfibulare"  # "Os subfibulare", "Ossikel lateral Fibula", "zusätzlicher Knochenkern lateral"
    ANDERE = "Andere"  # "anderes Ossikel", "seltenes akzessorisches Element", "weitere Variante"


class RetinakulumPathologie(Enum):
    """
    Pathologien der Retinacula (Sehnenhaltebänder).

    LLM-Extraktionshilfe: Suche nach "Retinaculum", "Halteband", "SPR", "IPR"
    kombiniert mit pathologischen Begriffen.
    """

    VERDICKT = (
        "Verdickt"  # "verdickt", "Verdickung des Retinaculums", "knotig verdickt"
    )
    OEDEMATOES = "Oedematoes"  # "ödematös", "Ödem im Retinaculum", "signalreich", "entzündlich verändert"
    PARTIALRUPTUR = "Partialruptur"  # "Partialruptur", "Teilruptur", "inkompletter Riss des Retinaculums"
    KOMPLETTRUPTUR = "Komplettruptur"  # "Komplettruptur", "Totalruptur", "durchtrennt", "kompletter Riss"
    POSTOPERATIV = "Postoperativ"  # "postoperativ", "Z.n. Operation", "rekonstruiert", "operativ versorgt"
    NICHT_ABGRENZBAR = "Nicht abgrenzbar"  # "nicht abgrenzbar", "schwer beurteilbar", "unklare Darstellung"


class PlantarfasziePathologie(Enum):
    """
    Pathologien der Plantarfaszie (plantare Aponeurose).

    LLM-Extraktionshilfe: Suche nach "Plantarfaszie", "plantare Aponeurose",
    "Fasziitis", "Fersensporn" kombiniert mit pathologischen Begriffen.
    """

    FASZIITIS = "Fasziitis"  # "Plantarfasziitis", "Fasziitis", "entzündliche Plantarfaszie", "Fersensporn-Syndrom"
    PARTIALRUPTUR = "Partialruptur"  # "Partialruptur", "Teilruptur", "inkompletter Riss der Plantarfaszie"
    KOMPLETTRUPTUR = "Komplettruptur"  # "Komplettruptur", "Totalruptur", "kompletter Riss", "durchtrennte Plantarfaszie"
    FIBROMATOSE = "Fibromatose"  # "Fibromatose", "Morbus Ledderhose", "plantare Fibromatosis", "knotige Verdickung"
    PERIFASZIALES_OEDEM = "Perifasziales Oedem"  # "perifasziales Ödem", "Ödem um die Plantarfaszie", "Schwellung plantar"
    OEDEM_FERSENPOLSTER = "Oedem des Fersenpolsters"  # "Ödem des Fersenpolsters", "Heel pad edema", "Schwellung Fersenfettpolster"


class NerventunnelBefund(Enum):
    """
    Pathologische Befunde im Bereich des Tarsaltunnels und anderer Nervendurchtrittsstellen.

    LLM-Extraktionshilfe: Suche nach "Tarsaltunnel", "Nervus tibialis", "Kompression",
    "Ganglion", "Raumforderung" im Bereich des medialen Knöchels.
    """

    KEIN = "Kein"  # Negativbefund: "kein Tarsaltunnelsyndrom", "freier Nervendurchtritt", "unauffälliger Tarsaltunnel"
    TARSALTUNNEL_NARBE = "Narben/Fibrose im Tarsaltunnel"  # "Narbenbildung", "Fibrose", "vernarbte Strukturen im Tarsaltunnel"
    GANGLION = "Ganglion mit Nervenkompression"  # "Ganglion", "Zyste mit Kompression", "raumfordernde Läsion"
    RAUMFORDERUNG = (
        "Raumforderung"  # "Raumforderung", "Tumor", "verdächtige Läsion", "Schwellung"
    )
    NEURITIS = (
        "Neuritis"  # "Neuritis", "Nervenentzündung", "entzündliche Nervenveränderung"
    )
    VARICOSIS = (
        "Varicosis"  # "Varikosis", "Krampfadern", "venöse Stauung", "Venenerweiterung"
    )


class WeichteilBefund(Enum):
    """
    Weichteilpathologien im Sprunggelenkbereich.

    LLM-Extraktionshilfe: Suche nach Weichteilvernänderungen, Ödemen,
    Hämatomen und spezifischen Syndromen.
    """

    OEDEM = "OEdem"  # "Ödem", "Schwellung", "Flüssigkeitsansammlung", "hyperintense Signalveränderung"
    HAEMATOM = "Haematom"  # "Hämatom", "Blutung", "Einblutung", "Blutansammlung"
    NARBE = "Narbe"  # "Narbe", "Vernarbung", "fibröse Veränderung", "Fibrose"
    POSTOPERATIV = "Postoperativ"  # "postoperativ", "Z.n. Operation", "operativ versorgt", "Operationsfolgen"
    SINUS_TARSI_SYNDROM = "Sinus-tarsi-Syndrom"  # "Sinus-tarsi-Syndrom", "Fibrose im Sinus tarsi", "Sinus tarsi Pathologie"


class Artefakt(Enum):
    """
    Bildgebungsartefakte die die MRT-Beurteilbarkeit beeinträchtigen.

    LLM-Extraktionshilfe: Suche nach "Artefakt", "Beurteilbarkeit", "eingeschränkt",
    "Bewegung", "Metall" in Zusammenhang mit Bildqualität.
    """

    BEWEGUNG = "Bewegungsartefakt"  # "Bewegungsartefakt", "Bewegungsunschärfe", "Motion artifact", "unruhiger Patient"
    METALL = "Metallartefakt"  # "Metallartefakt", "Suszeptibilitätsartefakt durch Metall", "Implantate", "ferromagnetische Störung"
    SUSZEPTIBILITAET = "Suszeptibilitaetsartefakt"  # "Suszeptibilitätsartefakt", "Susceptibility artifact", "Signalauslöschung", "Verzerrung"
    EINGESCHRAENKT = "Eingeschraenkte Beurteilbarkeit"  # "eingeschränkte Beurteilbarkeit", "limitierte Aussagekraft", "nicht beurteilbar", "technisch unzureichend"


class TalarDomeLokalisation(Enum):
    """
    Lokalisation von Läsionen am Talusdome (Trochlea tali).

    LLM-Extraktionshilfe: Suche nach "Talus", "Talusdome", "Trochlea"
    kombiniert mit Richtungsangaben (medial, lateral, anterior, posterior).
    """

    MEDIAL = (
        "Medial"  # "medial am Talus", "mediale Facette", "Innenseite des Talusdomes"
    )
    ZENTRAL = "Zentral"  # "zentral am Talus", "zentrale Läsion", "Mitte des Talusdomes"
    LATERAL = (
        "Lateral"  # "lateral am Talus", "laterale Facette", "Außenseite des Talusdomes"
    )
    ANTEROLATERAL = (
        "Anterolateral"  # "anterolateral", "vorne lateral", "antero-lateral am Talus"
    )
    POSTEROLATERAL = "Posterolateral"  # "posterolateral", "hinten lateral", "postero-lateral am Talus"
    ANTEROMEDIAL = (
        "Anteromedial"  # "anteromedial", "vorne medial", "antero-medial am Talus"
    )
    POSTEROMEDIAL = (
        "Posteromedial"  # "posteromedial", "hinten medial", "postero-medial am Talus"
    )
    NICHT_SPEZIFIZIERT = "Nicht spezifiziert"  # "unspezifische Lokalisation", "nicht näher lokalisiert", "Talus allgemein"


class DistaleTibiaLokalisation(Enum):
    """
    Lokalisation von Läsionen an der distalen Tibia.

    LLM-Extraktionshilfe: Suche nach "distale Tibia", "Tibia distal", "Pilon tibial"
    kombiniert mit Richtungsangaben (medial, lateral, zentral).
    """

    MEDIAL = "Medial"  # "medial an der distalen Tibia", "mediale Gelenkfläche", "Innenknöchel-nah"
    ZENTRAL = "Zentral"  # "zentral an der distalen Tibia", "zentrale Gelenkfläche", "Mitte der Tibia"
    LATERAL = "Lateral"  # "lateral an der distalen Tibia", "laterale Gelenkfläche", "außen an der Tibia"
    NICHT_SPEZIFIZIERT = "Nicht spezifiziert"  # "unspezifische Lokalisation", "distale Tibia allgemein", "nicht näher lokalisiert"


class Gelenkfluessigkeiten(BaseModel):
    OSG_ERGUSS: LabelWithOptionalReferences[GelenkergussGrad] = Field(
        description="Gelenkerguss im oberen Sprunggelenk (OSG). LLM soll nach 'Erguss', 'Flüssigkeit', 'Distension' im OSG suchen. Bei KEIN: keine Referenzen nötig."
    )
    USG_SUBTALARGELENK_ERGUSS: LabelWithOptionalReferences[GelenkergussGrad] = Field(
        description="Gelenkerguss im Subtalargelenk (hintere Facette des USG). LLM soll nach 'subtalarer Erguss', 'Flüssigkeit im Subtalargelenk' suchen."
    )
    USG_TALONAVICULARGELENK_ERGUSS: LabelWithOptionalReferences[GelenkergussGrad] = (
        Field(
            description="Gelenkerguss im Talonaviculargelenk (Teil des USG). LLM soll nach 'talonavikulärer Erguss', 'Flüssigkeit im Chopart-Gelenk' suchen."
        )
    )
    SYNOVITIS: LabelWithOptionalReferences[SynovitisGrad] = Field(
        description="Entzündung der Synovialmembran. LLM soll nach 'Synovitis', 'synoviale Proliferation', 'Enhancement der Synovia' suchen. Bei KEIN: keine Referenzen nötig."
    )
    ganglion_oder_kapselaussackung: LabelWithOptionalReferences[bool] = Field(
        description="Ganglion oder Kapselaussackung am Sprunggelenk. LLM soll nach 'Ganglion', 'Zyste', 'Kapselaussackung', 'Baker-Zyste' suchen. Bei false: keine Referenzen nötig."
    )
    freie_gelenkkoerper_chondral: LabelWithOptionalReferences[bool] = Field(
        description="Freie Gelenkkörper aus Knorpelmaterial. LLM soll nach 'freie Gelenkkörper', 'Knorpelfragment', 'loose body', 'chondrales Fragment' suchen. Bei false: keine Referenzen nötig."
    )
    freie_gelenkkoerper_osteochondral: LabelWithOptionalReferences[bool] = Field(
        description="Freie Gelenkkörper aus Knorpel-Knochen-Material. LLM soll nach 'osteochondraler freier Körper', 'Knochen-Knorpel-Fragment' suchen. Bei false: keine Referenzen nötig."
    )


class Aussenbandkomplex(BaseModel):
    LFTA: Optional[List[LabelWithReferences[Bandpathologie]]] = Field(
        default=None,
        description="Ligamentum fibulotalare anterius (vorderes Außenband). LLM soll nach 'LFTA', 'vorderes Außenband', 'Ligamentum fibulotalare anterius', 'fibulotalares Band anterior' suchen.",
    )
    LFC: Optional[List[LabelWithReferences[Bandpathologie]]] = Field(
        default=None,
        description="Ligamentum fibulocalcaneare (mittleres Außenband). LLM soll nach 'LFC', 'Ligamentum fibulocalcaneare', 'fibulocalcaneares Band', 'mittleres Außenband' suchen.",
    )
    LFTP: Optional[List[LabelWithReferences[Bandpathologie]]] = Field(
        default=None,
        description="Ligamentum fibulotalare posterius (hinteres Außenband). LLM soll nach 'LFTP', 'hinteres Außenband', 'Ligamentum fibulotalare posterius', 'fibulotalares Band posterior' suchen.",
    )
    SYNDESMOSE_ANTERIOR: Optional[List[LabelWithReferences[Bandpathologie]]] = Field(
        default=None,
        description="Vordere Syndesmose (anterior-tibiofibulares Band). LLM soll nach 'vordere Syndesmose', 'AITFL', 'anterior tibiofibular ligament', 'Syndesmose anterior' suchen.",
    )
    SYNDESMOSE_POSTERIOR: Optional[List[LabelWithReferences[Bandpathologie]]] = Field(
        default=None,
        description="Hintere Syndesmose (posterior-tibiofibulares Band). LLM soll nach 'hintere Syndesmose', 'PITFL', 'posterior tibiofibular ligament', 'Syndesmose posterior' suchen.",
    )
    INTEROSSAERES_BAND: Optional[List[LabelWithReferences[Bandpathologie]]] = Field(
        default=None,
        description="Interossäres Band der Syndesmose. LLM soll nach 'interossäres Band', 'Membrana interossea', 'IOL', 'interossäre Membran' suchen.",
    )


class Weiterebaender(BaseModel):
    LIGAMENTUM_BIFURCATUM: Optional[List[LabelWithReferences[Bandpathologie]]] = Field(
        default=None,
        description="Ligamentum bifurcatum (Y-Band). LLM soll nach 'Ligamentum bifurcatum', 'Y-Band', 'bifurcates Ligament' am Calcaneus-Naviculare/Cuboid suchen.",
    )
    LIGAMENTUM_INTERMALLEOLARE_POSTERIUS: Optional[
        List[LabelWithReferences[Bandpathologie]]
    ] = Field(
        default=None,
        description="Ligamentum intermalleolare posterius. LLM soll nach 'Ligamentum intermalleolare posterius', 'hinteres intermalleoläres Band' zwischen den Malleoli suchen.",
    )
    LIGAMENTUM_TALONAVICULARE_DORSALE: Optional[
        List[LabelWithReferences[Bandpathologie]]
    ] = Field(
        default=None,
        description="Ligamentum talonaviculare dorsale (dorsales talonavikuläres Band). LLM soll nach 'talonavikuläres Band', 'dorsales Band Talus-Naviculare' suchen.",
    )


class Innenbandkomplex(BaseModel):
    OBERFLAECHLICHES_INNENBAND: Optional[List[LabelWithReferences[Bandpathologie]]] = (
        Field(
            default=None,
            description="Oberflächliches Deltaband (Ligamentum deltoideum superficiale). LLM soll nach 'oberflächliches Deltaband', 'Ligamentum deltoideum', 'superficial deltoid ligament', 'Innenband oberflächlich' suchen.",
        )
    )
    TIEFES_INNENBAND: Optional[List[LabelWithReferences[Bandpathologie]]] = Field(
        default=None,
        description="Tiefes Deltaband (Ligamentum deltoideum profundum). LLM soll nach 'tiefes Deltaband', 'deep deltoid ligament', 'Ligamentum deltoideum profundum', 'Innenband tief' suchen.",
    )
    SPRING_LIGAMENT: Optional[List[LabelWithReferences[Bandpathologie]]] = Field(
        default=None,
        description="Spring-Ligament (Ligamentum calcaneonaviculare plantare). LLM soll nach 'Spring-Ligament', 'Ligamentum calcaneonaviculare', 'plantares calcaneonavikuläres Band' suchen.",
    )


class Retinakula(BaseModel):
    SUPERIORES_PERONEALES_RETINACULUM: Optional[
        List[LabelWithReferences[RetinakulumPathologie]]
    ] = Field(
        default=None,
        description="Superiores peroneales Retinaculum (oberes Peronealsehnenhalteb). LLM soll nach 'superiores Retinaculum', 'oberes peroneales Halteband', 'SPR' suchen.",
    )
    INFERIORES_PERONEALES_RETINACULUM: Optional[
        List[LabelWithReferences[RetinakulumPathologie]]
    ] = Field(
        default=None,
        description="Inferiores peroneales Retinaculum (unteres Peronealsehnenhalteb). LLM soll nach 'inferiores Retinaculum', 'unteres peroneales Halteband', 'IPR' suchen.",
    )
    EXTENSOR_RETINACULUM: Optional[List[LabelWithReferences[RetinakulumPathologie]]] = (
        Field(
            default=None,
            description="Extensor Retinaculum (Strecksehnenhalteb). LLM soll nach 'Extensor Retinaculum', 'Strecksehnenhalteb', 'dorsales Halteband' über den Sehnen suchen.",
        )
    )
    FLEXOR_RETINACULUM: Optional[List[LabelWithReferences[RetinakulumPathologie]]] = (
        Field(
            default=None,
            description="Flexor Retinaculum (Tarsaltunnel-Dach). LLM soll nach 'Flexor Retinaculum', 'Beugesehnenhalteb', 'Tarsaltunnel', 'Laciniatum' suchen.",
        )
    )


class Peronealsehnen(BaseModel):
    PERONEUS_BREVIS: Optional[List[LabelWithReferences[Sehnenpathologie]]] = Field(
        default=None,
        description="Peroneus brevis Sehne (kurzer Wadenbeinmuskel). LLM soll nach 'Peroneus brevis', 'Fibularis brevis', 'kurzer Peroneus' lateral am Sprunggelenk suchen.",
    )
    PERONEUS_LONGUS: Optional[List[LabelWithReferences[Sehnenpathologie]]] = Field(
        default=None,
        description="Peroneus longus Sehne (langer Wadenbeinmuskel). LLM soll nach 'Peroneus longus', 'Fibularis longus', 'langer Peroneus' lateral am Sprunggelenk suchen.",
    )
    tenosynovitis_grad: LabelWithOptionalReferences[TenosynovitisGrad] = Field(
        description="Tenosynovitis-Grad der Peronealsehnen. LLM soll nach 'Tenosynovitis', 'Sehnenscheidenentzündung', 'peritendinöses Ödem' bei Peronealsehnen suchen. Bei KEIN: keine Referenzen nötig."
    )


class MedialeFlexorensehnen(BaseModel):
    TIBIALIS_POSTERIOR: Optional[List[LabelWithReferences[Sehnenpathologie]]] = Field(
        default=None,
        description="Tibialis posterior Sehne (hinterer Schienbeinmuskel). LLM soll nach 'Tibialis posterior', 'M. tibialis posterior', 'PTTD' (Posterior Tibial Tendon Dysfunction) medial suchen.",
    )
    FLEXOR_HALLUCIS_LONGUS: Optional[List[LabelWithReferences[Sehnenpathologie]]] = (
        Field(
            default=None,
            description="Flexor hallucis longus Sehne (langer Großzehenbeuger). LLM soll nach 'Flexor hallucis longus', 'FHL', 'Großzehenbeuger' medial-plantar suchen.",
        )
    )
    FLEXOR_DIGITORUM_LONGUS: Optional[List[LabelWithReferences[Sehnenpathologie]]] = (
        Field(
            default=None,
            description="Flexor digitorum longus Sehne (langer Zehenbeuger). LLM soll nach 'Flexor digitorum longus', 'FDL', 'Zehenbeuger' medial suchen.",
        )
    )
    tenosynovitis_grad: LabelWithOptionalReferences[TenosynovitisGrad] = Field(
        description="Tenosynovitis-Grad der medialen Flexorensehnen. LLM soll nach 'Tenosynovitis', 'Sehnenscheidenentzündung' bei medialen Sehnen suchen. Bei KEIN: keine Referenzen nötig."
    )


class Extensorensehnen(BaseModel):
    TIBIALIS_ANTERIOR: Optional[List[LabelWithReferences[Sehnenpathologie]]] = Field(
        default=None,
        description="Tibialis anterior Sehne (vorderer Schienbeinmuskel). LLM soll nach 'Tibialis anterior', 'M. tibialis anterior', 'vorderer Schienbeinmuskel' dorsal suchen.",
    )
    EXTENSOR_HALLUCIS_LONGUS: Optional[List[LabelWithReferences[Sehnenpathologie]]] = (
        Field(
            default=None,
            description="Extensor hallucis longus Sehne (langer Großzehenstrecker). LLM soll nach 'Extensor hallucis longus', 'EHL', 'Großzehenstrecker' dorsal suchen.",
        )
    )
    EXTENSOR_DIGITORUM_LONGUS: Optional[List[LabelWithReferences[Sehnenpathologie]]] = (
        Field(
            default=None,
            description="Extensor digitorum longus Sehne (langer Zehenstrecker). LLM soll nach 'Extensor digitorum longus', 'EDL', 'Zehenstrecker' dorsal suchen.",
        )
    )
    tenosynovitis_grad: LabelWithOptionalReferences[TenosynovitisGrad] = Field(
        description="Tenosynovitis-Grad der Extensorensehnen. LLM soll nach 'Tenosynovitis', 'Sehnenscheidenentzündung' bei dorsalen Sehnen suchen. Bei KEIN: keine Referenzen nötig."
    )


class Achillessehne(BaseModel):
    ACHILLESSEHNE: Optional[List[LabelWithReferences[Sehnenpathologie]]] = Field(
        default=None,
        description="Achillessehne (Tendo calcaneus). LLM soll nach 'Achillessehne', 'Tendo Achillis', 'Tendo calcaneus', 'Achilles tendon' posterior suchen.",
    )
    PLANTARISSEHNE: Optional[List[LabelWithReferences[Sehnenpathologie]]] = Field(
        default=None,
        description="Plantarissehne (dünne Hilfsmuskel). LLM soll nach 'Plantarissehne', 'M. plantaris', 'Plantaris tendon' medial der Achillessehne suchen.",
    )
    zusatzbefunde: Optional[List[LabelWithReferences[AchillesBefund]]] = Field(
        default=None,
        description="Zusatzbefunde im Achillessehnenbereich. LLM soll nach spezifischen Achilles-Pathologien wie 'Haglund', 'Bursitis', 'Enthesiopathie' suchen.",
    )


class Plantarfaszie(BaseModel):
    pathologie: Optional[LabelWithReferences[PlantarfasziePathologie]] = Field(
        default=None,
        description="Pathologie der Plantarfaszie (Plantarfasziitis, etc.). LLM soll nach 'Plantarfaszie', 'Plantarfasziitis', 'plantare Aponeurose', 'Fasziitis' plantar suchen.",
    )


class OsteochondraleLaesion(BaseModel):
    lokalisation_talus: Optional[LabelWithReferences[TalarDomeLokalisation]] = Field(
        default=None,
        description="Lokalisation der osteochondralen Läsion am Talus. LLM soll nach 'Talus', 'medial/lateral/zentral am Talus', 'Talusdome' mit Richtungsangaben suchen.",
    )
    lokalisation_distale_tibia: Optional[
        LabelWithReferences[DistaleTibiaLokalisation]
    ] = Field(
        default=None,
        description="Lokalisation der osteochondralen Läsion an der distalen Tibia. LLM soll nach 'distale Tibia', 'Tibia distal', 'Pilon tibial' mit Richtungsangaben suchen.",
    )
    groesse_mm_mediolateral: Optional[LabelWithReferences[float]] = Field(
        default=None,
        description="Größe der Läsion in mm (mediolateral). LLM soll nach Größenangaben 'x mm mediolateral', 'Breite', 'Durchmesser' suchen.",
    )
    groesse_mm_anteroposterior: Optional[LabelWithReferences[float]] = Field(
        default=None,
        description="Größe der Läsion in mm (anteroposterior). LLM soll nach Größenangaben 'x mm anteroposterior', 'Länge', 'AP-Durchmesser' suchen.",
    )
    stabilitaet: LabelWithOptionalReferences[OsteochondraleStabilitaet] = Field(
        description="Stabilität der osteochondralen Läsion. LLM soll nach 'stabil', 'instabil', 'bewegliches Fragment', 'loose body' suchen. Bei STABIL oft keine explizite Referenz."
    )
    subchondrale_zyste: LabelWithOptionalReferences[bool] = Field(
        description="Vorhandensein subchondraler Zysten. LLM soll nach 'subchondrale Zyste', 'Geröllzyste', 'zystische Läsion' suchen. Bei false: keine Referenzen nötig."
    )
    subchondrales_oedem: LabelWithOptionalReferences[bool] = Field(
        description="Vorhandensein subchondraler Ödeme. LLM soll nach 'subchondrales Ödem', 'Knochenmarködem', 'STIR-hyperintens' suchen. Bei false: keine Referenzen nötig."
    )
    disloziertes_fragment: LabelWithOptionalReferences[bool] = Field(
        description="Vorhandensein dislozierter Fragmente. LLM soll nach 'disloziertes Fragment', 'verschobenes Fragment', 'freier Körper' suchen. Bei false: keine Referenzen nötig."
    )


class TalokruralerKnorpel(BaseModel):
    distale_Tibia: Optional[List[LabelWithReferences[Knorpelpathologie]]] = Field(
        default=None,
        description="Knorpelpathologien an der distalen Tibia im OSG. LLM soll nach 'Knorpel', 'chondral', 'Tibia distal', 'Gelenkfläche Tibia' suchen.",
    )
    talus: Optional[List[LabelWithReferences[Knorpelpathologie]]] = Field(
        default=None,
        description="Knorpelpathologien am Talus im OSG. LLM soll nach 'Knorpel', 'chondral', 'Talus', 'Talusdome', 'Trochlea tali' suchen.",
    )
    osteochondrale_laesion: Optional[OsteochondraleLaesion] = Field(
        default=None,
        description="Osteochondrale Läsionen im OSG. LLM soll nach 'osteochondrale Läsion', 'OCD', 'Knorpel-Knochen-Defekt', 'osteochondraler Defekt' suchen.",
    )


class SubtalarerKnorpel(BaseModel):
    hintere_facette: Optional[List[LabelWithReferences[Knorpelpathologie]]] = Field(
        default=None,
        description="Knorpelpathologien an der hinteren Facette des Subtalargelenks. LLM soll nach 'hintere Facette', 'posterior Facette', 'Subtalargelenk hinten' suchen.",
    )
    mittlere_facette: Optional[List[LabelWithReferences[Knorpelpathologie]]] = Field(
        default=None,
        description="Knorpelpathologien an der mittleren Facette des Subtalargelenks. LLM soll nach 'mittlere Facette', 'middle Facette', 'Subtalargelenk mitte' suchen.",
    )
    vordere_facette: Optional[List[LabelWithReferences[Knorpelpathologie]]] = Field(
        default=None,
        description="Knorpelpathologien an der vorderen Facette des Subtalargelenks. LLM soll nach 'vordere Facette', 'anterior Facette', 'Subtalargelenk vorne' suchen.",
    )


class KnochenBefunde(BaseModel):
    tibia_distal: Optional[List[LabelWithReferences[Knochenpathologie]]] = Field(
        default=None,
        description="Knochenpathologien an der distalen Tibia. LLM soll nach 'Tibia distal', 'distale Tibia', 'Pilon tibial', 'Schienbein distal' suchen.",
    )
    fibula_distal: Optional[List[LabelWithReferences[Knochenpathologie]]] = Field(
        default=None,
        description="Knochenpathologien an der distalen Fibula (Außenknöchel). LLM soll nach 'Fibula distal', 'Außenknöchel', 'Malleolus lateralis', 'Wadenbein distal' suchen.",
    )
    talus: Optional[List[LabelWithReferences[Knochenpathologie]]] = Field(
        default=None,
        description="Knochenpathologien am Talus (Sprungbein). LLM soll nach 'Talus', 'Sprungbein', 'Talusdome', 'Trochlea tali' suchen.",
    )
    calcaneus: Optional[List[LabelWithReferences[Knochenpathologie]]] = Field(
        default=None,
        description="Knochenpathologien am Calcaneus (Fersenbein). LLM soll nach 'Calcaneus', 'Fersenbein', 'Tuber calcanei' suchen.",
    )
    naviculare: Optional[List[LabelWithReferences[Knochenpathologie]]] = Field(
        default=None,
        description="Knochenpathologien am Os naviculare (Kahnbein). LLM soll nach 'Naviculare', 'Os naviculare', 'Kahnbein' suchen.",
    )
    cuboid: Optional[List[LabelWithReferences[Knochenpathologie]]] = Field(
        default=None,
        description="Knochenpathologien am Os cuboideum (Würfelbein). LLM soll nach 'Cuboid', 'Os cuboideum', 'Würfelbein' suchen.",
    )


class AlignmentInstabilitaet(BaseModel):
    Rueckfussstellung: LabelWithOptionalReferences[_Rueckfussstellung] = Field(
        description="Rückfußstellung/Alignment. LLM soll nach 'Rückfuß', 'Alignment', 'Achse', 'Stellung', 'Valgus', 'Varus' suchen. Bei NEUTRAL: oft keine explizite Referenz."
    )
    syndesmose_weit: LabelWithOptionalReferences[bool] = Field(
        description="Syndesmosenerweiterung. LLM soll nach 'Syndesmose weit', 'erweiterte Syndesmose', 'Syndesmosenverletzung', 'Diastase' suchen. Bei false: keine Referenzen nötig."
    )
    chronische_laterale_instabilitaet: LabelWithOptionalReferences[bool] = Field(
        description="Chronische laterale Instabilität. LLM soll nach 'laterale Instabilität', 'chronische Instabilität', 'Außenbandinsuffizienz' suchen. Bei false: keine Referenzen nötig."
    )
    mediale_instabilitaet: LabelWithOptionalReferences[bool] = Field(
        description="Mediale Instabilität. LLM soll nach 'mediale Instabilität', 'Innenbandinsuffizienz', 'Deltabandläsion' suchen. Bei false: keine Referenzen nötig."
    )


class OssikelVarianten(BaseModel):
    vorhanden: Optional[List[LabelWithReferences[AkzessorischesOssikel]]] = Field(
        default=None,
        description="Vorhandene akzessorische Ossikeln. LLM soll nach 'Os trigonum', 'Os peroneum', 'akzessorisches Ossikel', 'Stieda-Fortsatz' suchen.",
    )


class NeurovaskulaerWeichteile(BaseModel):
    tarsaltunnel: LabelWithOptionalReferences[NerventunnelBefund] = Field(
        description="Tarsaltunnel-Befunde. LLM soll nach 'Tarsaltunnel', 'Nervus tibialis', 'Kompression', 'Ganglion medial' suchen. Bei KEIN: keine Referenzen nötig."
    )
    weichteile: Optional[List[LabelWithReferences[WeichteilBefund]]] = Field(
        default=None,
        description="Weichteilbefunde. LLM soll nach 'Ödem', 'Hämatom', 'Narbe', 'postoperativ', 'Sinus-tarsi-Syndrom' in Weichteilen suchen.",
    )
    subkutanes_oedem: LabelWithOptionalReferences[bool] = Field(
        description="Subkutanes Ödem. LLM soll nach 'subkutanes Ödem', 'Schwellung', 'Hautschichtenödem' suchen. Bei false: keine Referenzen nötig."
    )
    varicosis_oder_venenverdickung: LabelWithOptionalReferences[bool] = Field(
        description="Varikosis oder Venenverdickung. LLM soll nach 'Varikosis', 'Krampfadern', 'Venenerweiterung', 'venöse Stauung' suchen. Bei false: keine Referenzen nötig."
    )


class Postoperativ(BaseModel):
    metallimplantate: LabelWithOptionalReferences[bool] = Field(
        description="Vorhandensein von Metallimplantaten. LLM soll nach 'Metallimplantate', 'Schrauben', 'Platten', 'Metallartefakte' suchen. Bei false: keine Referenzen nötig."
    )
    schraubenkanaele: LabelWithOptionalReferences[bool] = Field(
        description="Schraubenkanäle von entfernten Implantaten. LLM soll nach 'Schraubenkanäle', 'ehemalige Schrauben', 'Bohrkanäle' suchen. Bei false: keine Referenzen nötig."
    )
    osteotomiezeichen: LabelWithOptionalReferences[bool] = Field(
        description="Zeichen durchgeführter Osteotomien. LLM soll nach 'Osteotomie', 'Z.n. Knochendurchtrennung', 'Osteotomiezeichen' suchen. Bei false: keine Referenzen nötig."
    )


class UnrecognizedEntity(BaseModel):
    unbekannte_diagnose: str = Field(
        description="Medizinische Diagnose oder anatomische Befund, der nicht im Schema erfasst ist"
    )
    textabschnitt: str = Field(description="Originaltext aus dem MRT-Befund")
    vorgeschlagenes_feld: str = Field(description="Name für ein neues Schema-Feld")


class SprunggelenkMRTBericht(BaseModel):
    gelenkfluessigkeiten: Gelenkfluessigkeiten = Field(
        description="Gelenkflüssigkeiten, Ergüsse und synoviale Befunde in OSG und USG"
    )
    aussenbandkomplex: Aussenbandkomplex = Field(
        description="Außenbandkomplex mit LFTA, LFC, LFTP und Syndesmose"
    )
    innenbandkomplex: Innenbandkomplex = Field(
        description="Innenbandkomplex mit Deltaband und Spring-Ligament"
    )
    retinakula: Retinakula = Field(
        description="Sehnenhaltebänder (Retinacula) für Peronealsehnen, Extensoren und Flexoren"
    )
    peronealsehnen: Peronealsehnen = Field(
        description="Peronealsehnen (Peroneus brevis/longus) und deren Tenosynovitis"
    )
    mediale_flexorensehnen: MedialeFlexorensehnen = Field(
        description="Mediale Flexorensehnen (Tibialis posterior, FHL, FDL) und deren Tenosynovitis"
    )
    extensorensehnen: Extensorensehnen = Field(
        description="Extensorensehnen (Tibialis anterior, EHL, EDL) und deren Tenosynovitis"
    )
    achillessehne: Achillessehne = Field(
        description="Achillessehne, Plantarissehne und spezifische Achilles-Pathologien"
    )
    plantarfaszie: Plantarfaszie = Field(
        description="Plantarfaszie und deren Pathologien (Fasziitis, etc.)"
    )
    talokruraler_knorpel: TalokruralerKnorpel = Field(
        description="Knorpel des oberen Sprunggelenks (Talus und distale Tibia)"
    )
    subtalarer_knorpel: SubtalarerKnorpel = Field(
        description="Knorpel des Subtalargelenks (hintere, mittlere, vordere Facette)"
    )
    knochen_befunde: KnochenBefunde = Field(
        description="Knochenpathologien aller relevanten Fußwurzelknochen"
    )
    alignment_instabilitaet: AlignmentInstabilitaet = Field(
        description="Ausrichtung und Instabilitäten des Sprunggelenks"
    )
    ossikel_varianten: OssikelVarianten = Field(
        description="Akzessorische Ossikel und anatomische Varianten"
    )
    neurovaskulaer_weichteile: NeurovaskulaerWeichteile = Field(
        description="Neurovaskuläre Strukturen, Tarsaltunnel und Weichteile"
    )
    postoperativ: Optional[Postoperativ] = Field(
        default=None,
        description="Postoperative Befunde wie Implantate und Osteotomiezeichen",
    )
    artefakte: Optional[List[LabelWithReferences[Artefakt]]] = Field(
        default=None,
        description="Bildgebungsartefakte die die Beurteilbarkeit einschränken",
    )
    zn_operation: LabelWithOptionalReferences[bool] = Field(
        description="Z.n. Operation am Sprunggelenk in der Anamnese. LLM soll nach 'Z.n. Operation', 'postoperativ', 'operiert' suchen."
    )
    zn_sprunggelenksdistorsion: LabelWithOptionalReferences[bool] = Field(
        description="Z.n. Sprunggelenksdistorsion in der Anamnese. LLM soll nach 'Distorsion', 'Umknicken', 'Verstauchung', 'Trauma' suchen."
    )

    # Unified list for schema gaps
    unbekannte_diagnosen: List[UnrecognizedEntity] = Field(
        default_factory=list,
        description="Informationen/Diagnosen, die nicht in das neue Schema passen",
    )
