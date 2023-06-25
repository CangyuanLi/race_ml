# Imports

from __future__ import annotations

import datetime
from collections.abc import Sequence
from pathlib import Path

import cutils
import polars as pl
import spacy
import tqdm

from utils.constants import VALID_NAME_CHARS
from utils.paths import CENSUS_PATH, CW_PATH, FINAL_PATH, PPP_PATH

NLP = spacy.load("en_core_web_sm")

INVALID_NAMES_ANY = {
    "&",
    ",",
    "academy",
    "account",
    "accounting",
    "clothing",
    "redemption",
    "parkside",
    "dynasty",
    "dispensaries",
    "dispensary",
    "hankookbbqhouseinc",
    "multimedia",
    "northeast",
    "twisted",
    "irrigation",
    "children",
    "borderline",
    "chevrolet",
    "greenhouse",
    "seashore",
    "riverboat",
    "resource",
    "independent",
    "nurseries",
    "acquisition",
    "administration",
    "administrative",
    "embroider",
    "cuisine",
    "adoption",
    "advance",
    "advisor",
    "pharmacies",
    "shellfish",
    "shelter",
    "aerospace",
    "dynamic",
    "mortuary",
    "caterer",
    "avenue",
    "gallery",
    "treatment",
    "cambridge",
    "behavioral",
    "atlantic",
    "aesthetic",
    "affordable",
    "agency",
    "agricultur",
    "alliance",
    "alternative",
    "ambulance",
    "analytics",
    "anesthesia",
    "anesthesiology",
    "corportaion",
    "pancake",
    "boomerang",
    "augusta eye",
    "anglican",
    "appliance",
    "applied",
    "appraisal",
    "architect",
    "architectural",
    "architecture",
    "asphalt",
    "assisted",
    "associate",
    "association",
    "athletic",
    "attraction",
    "auction",
    "audience",
    "authority",
    "automatic",
    "automotive",
    "baptist",
    "barbecue",
    "barbershop",
    "bodywork",
    "boutique",
    "brethren",
    "broker",
    "building",
    "business",
    "cardiology",
    "cardiovascular",
    "carpenter",
    "catering",
    "catholic church",
    "catholic",
    "challenge",
    "charitable",
    "chemist",
    "childcare",
    "chiropractic",
    "christian church",
    "classroom",
    "cleaners",
    "cleaning",
    "coalition",
    "collection",
    "collective",
    "collision",
    "combination",
    "commercial",
    "commission",
    "communication",
    "communities",
    "community",
    "companies",
    "company",
    "component",
    "composer",
    "computer",
    "concenssion",
    "clear pool",
    "concession",
    "concrete",
    "conditioning",
    "condominium",
    "conductor",
    "confection",
    "conference",
    "congregation",
    "connection",
    "conservancy",
    "conservation",
    "consortium",
    "construction",
    "constructor",
    "consultant",
    "consulting",
    "contracting",
    "contractor",
    "control",
    "cooperative",
    "corporate",
    "corporation",
    "corporaton",
    "corportation",
    "corpotation",
    "cosmetic",
    "counseling",
    "covenant",
    "custom",
    "day care",
    "daycare",
    "dentist",
    "dermatology",
    "design",
    "development",
    "diagnostic",
    "dimension",
    "distribution",
    "district",
    "domestic",
    "autism",
    "intervention",
    "automate",
    "excavat",
    "laborator",
    "corporatoion",
    "organic",
    "bakery",
    "drywall",
    "education",
    "electric",
    "electronic",
    "elementary",
    "emergency",
    "empower",
    "endeavor",
    "engineer",
    "enrichment",
    "enterprise",
    "entertainment",
    "environment",
    "episcopal",
    "equipment",
    "estate",
    "evangelical",
    "excavation",
    "excellence",
    "exclusive",
    "executive",
    "experience",
    "expression",
    "extraordinaire",
    "fabrication",
    "fellowship",
    "financial",
    "flooring",
    "florida",
    "focused",
    "foundation",
    "funeral",
    "furniture",
    "gastroenterology",
    "geriatric",
    "glacier",
    "radiation",
    "oncology",
    "global",
    "government",
    "gravestone",
    "gymnastic",
    "gynecology",
    "handyman",
    "hardware",
    "hardwood",
    "harvesting",
    "healthcare",
    "heating",
    "holding",
    "holiday",
    "homestead",
    "honeymoon",
    "hospice",
    "hospitality",
    "hydraulic",
    "immaculate",
    "immigration",
    "improvement",
    "incorporated",
    "incredible",
    "individual",
    "industrial",
    "industries",
    "ingredient",
    "initiative",
    "injection",
    "innovation",
    "inspection",
    "installation",
    "institute",
    "insulation",
    "insurance",
    "integrated",
    "intellectual",
    "intelligence",
    "interior",
    "international",
    "investment",
    "irrevocable",
    "jewelry",
    "kitchen",
    "l l c",
    "laboratorios",
    "laboratory",
    "landmark",
    "landscape",
    "landscaping",
    "leadership",
    "learning",
    "license",
    "limited",
    "campfire",
    "interactive",
    "sculpting",
    "mineral",
    "livestock",
    "locksmith",
    "logistic",
    "lutheran",
    "machine",
    "madness",
    "maintenace",
    "maintenance",
    "maintenance",
    "makeup",
    "management",
    "manager",
    "manufacturing",
    "marketing",
    "massage",
    "mattress",
    "mechanical",
    "medical",
    "medicine",
    "memorial",
    "metabolic",
    "methodist",
    "mexican",
    "midnight",
    "ministries",
    "ministry",
    "missionary",
    "mitigation",
    "molecular",
    "mortgage",
    "mountain",
    "museum",
    "network",
    "neurology",
    "nursery",
    "nursing",
    "operation",
    "opportunities",
    "opportunity",
    "optometric",
    "optometry",
    "orchestra",
    "ordinary",
    "organization",
    "oriental",
    "orthodontics",
    "outlet",
    "packaging",
    "painting",
    "panhandle",
    "parenthood",
    "partner",
    "partnership",
    "paving",
    "payroll",
    "pediatric",
    "peninsula",
    "pentecostal",
    "perfection",
    "perform",
    "permanent",
    "petroleum",
    "pharmacy",
    "photography",
    "physical",
    "physician",
    "pizzeria",
    "placement",
    "plumbing",
    "precision",
    "presbyterian",
    "preschool",
    "preservation",
    "primary",
    "processing",
    "procurement",
    "product",
    "production",
    "professional",
    "profit",
    "program",
    "promotion",
    "promotional",
    "properties",
    "protection",
    "psychiatry",
    "publication",
    "publish",
    "quality",
    "radiologist",
    "radiology",
    "realty",
    "recovery",
    "recycling",
    "refrigeration",
    "rehabilitation",
    "relationship",
    "removal",
    "research",
    "residence",
    "residential",
    "restaurant",
    "restoration",
    "roofing",
    "sanitation",
    "school",
    "science",
    "seafood",
    "service",
    "society",
    "software",
    "solution",
    "southwestern",
    "specialist",
    "specialize",
    "specialties",
    "specialty",
    "staffing",
    "stainless",
    "stampede",
    "steakhouse",
    "storage",
    "strategies",
    "supermarket",
    "surgeon",
    "surgery",
    "suspend",
    "synagogue",
    "system",
    "taekwondo",
    "technical",
    "technician",
    "technique",
    "technologies",
    "technology",
    "theological",
    "therapeutic",
    "therapy",
    "cafeteria",
    "emancipation",
    "expectation",
    "federation",
    "birmingham",
    "sacrament",
    "anesthesiolog",
    "exterminat",
    "broadcast",
    "together",
    "transformation",
    "transmission",
    "transmission",
    "transport",
    "transportation",
    "trucking",
    "underground",
    "university",
    "unlimited",
    "vacation",
    "venture",
    "veteran",
    "veterinary",
    "vineyard",
    "vision care",
    "visiting",
    "visual",
    "volunteer",
    "cookie dough",
    "sheshedartstudio",
    "mae trucks",
    "art prize",
    "embroider",
    "servicios",
    "warehouse",
    "welding",
    "wellness",
    "centro de  terapia fisica y clinica del dolor csp",
    "workshop",
    "entperprise",
    "delicatessen",
    "boston",
    "wesleyan church",
    "wholesale",
    "window",
    "wireless",
    "woodwork",
    "workout",
    "worldwide",
    "boundary",
    "bowling",
    "import",
    "operating",
    "builder",
    "momentum",
    "worship",
    "xtreme",
    "growing",
    "sensation",
    "orthodox",
    "horticulture",
    "lawncare",
    "basketball",
    "football",
    "chemical",
    "graphic",
    "basement",
    "limestone",
    "telephone",
    "kingdom",
    "distributor",
    "ale house",
    "material",
    "lodging",
    "brazilian",
    "ice cream",
    "coporation",
    "optical",
    "awesome",
    "rental",
    "uniform",
    "surgical",
    "safety",
    "scanner",
    "grinding",
    "ophthalmology",
    "lawnscape",
    "evolution",
    "jeweler",
    "newport",
    "element",
    "motorsport",
    "barricade",
    "surface",
    "demolition",
    "small giants",
    "heartmath",
    "investigative",
    "botanical",
    "renovation",
    "eye care",
    "furnishing",
    "subway",
    "opticare",
    "noodle",
    "podiatry",
    "fabricator",
    "chicken",
    "saint matthew's church",
    "outdoor",
    "corproration",
    "baseball",
    "recreation",
    "florist",
    "vertical",
    "fitness",
    "doghouse",
    "critical",
    "plastic",
    "intangible",
    "treasure",
    "seating",
    "camera",
    "private",
    "security",
    "'s",
    "discovery",
    "tansportation",
    "orthopedic",
    "printing",
    "pioneer",
    "shoemaker",
    "brother",
    "elevator",
    "venue",
    "attorneys",
    "commissary",
    "skincare",
    "seminary",
    "construciton",
    "library",
    "general",
    "providence",
    "cathedral",
    "trailer",
    "urology",
    "fiberglass",
    "respiratory",
    "multicare",
    "massoninc",
    "crossroad",
    "investigation",
    "imaging",
    "provision",
    "buick gmc",
    "apothecary",
    "wealthy",
    "metal",
    "ymca",
    "dunkin",
    "mgmt",
    "surveyor",
    "town square",
    "corportion",
    "masonry",
    "reataurant",
    "manufactoring",
    "employment",
    "tropical",
    "legacy",
    "creation",
    "apartment",
    "distribut",
    "acupunctur",
    "acupuntcure",
    "accounitng",
    "accessor",
    "ladyfinger",
    "rainbow",
    "corporatiion",
    "compute",
    "concept",
    "wetpaint",
    "alteration",
    "nationx",
    "common",
    "d-hairs",
    "ceogroup",
    "foodle",
    "practical",
    "needlework",
    "pressure",
    "stylemaster",
    "greenlux",
    "hotmail",
    "dependable",
    "elegant",
    "granville",
    "towing",
    "trademaster",
    "barefoot",
    "taskrabbit",
    "iconstaff",
    "a-team",
    "a-z",
    "therapist",
    "therapy",
    "therapies",
    "threading",
    "installer",
    "survey",
    "laundromat",
    "take-two",
    "expected",
    "assistance",
    "attachment",
    "impression",
    "incorported",
    "influence",
    "janitorial",
    "paragon",
    "exploration",
    "airport",
    "contract",
    "sportfish",
    "restuarant",
    "developer",
    "creative",
    "sportswea",
    "homecare",
}

INVALID_NAMES_WORD = {
    "cabinets",
    "repast",
    "lines",
    "pbc",
    "laundry",
    "mfg",
    "modern",
    "ink",
    "inn",
    "intl",
    "sassy",
    "wash",
    "fans",
    "it",
    "sh",
    "image",
    "images",
    "uber",
    "true",
    "garage",
    "educare",
    "dc",
    "dr",
    "tlc",
    "gutter",
    "nail",
    "talk",
    "acts",
    "st",
    "condo",
    "condos",
    "ac",
    "lazy",
    "skm",
    "fashion",
    "care",
    "nd",
    "jl",
    "lpc",
    "on",
    "working",
    "pro",
    "wr",
    "wt",
    "la",
    "travel",
    "donuts",
    "md",
    "self",
    "nails",
    "casino",
    "resort",
    "vista",
    "yacht",
    "policy",
    "book",
    "books",
    "cente",
    "credit",
    "coast",
    "senior",
    "garden",
    "gardens",
    "famous",
    "motor",
    "parish",
    "legal",
    "lodge",
    "motors",
    "thrift",
    "store",
    "stores",
    "palace",
    "street",
    "iinc",
    "elevate",
    "usa",
    "media",
    "deli",
    "winery",
    "future",
    "american",
    "and",
    "island",
    "produce",
    "league",
    "truck",
    "motor",
    "archive",
    "assoc",
    "auto",
    "bbq",
    "cafe",
    "lc",
    "food",
    "studio",
    "studios",
    "foods",
    "beauty",
    "by",
    "center",
    "centers",
    "clinic",
    "co",
    "college",
    "corp",
    "youth",
    "council",
    "country",
    "county",
    "dental",
    "grill",
    "family",
    "farm",
    "farms",
    "club",
    "firm",
    "for",
    "group",
    "health",
    "hospital",
    "in",
    "inc",
    "lab",
    "labs",
    "liquor",
    "llc",
    "lllp",
    "llp",
    "lp",
    "ltd",
    "market",
    "metal",
    "nation",
    "national",
    "of",
    "office",
    "offices",
    "pizza",
    "plc",
    "pllc",
    "project",
    "public",
    "repair",
    "rescue",
    "salon",
    "school",
    "shop",
    "local",
    "supply",
    "spa",
    "studio",
    "tech",
    "the",
    "to",
    "towing",
    "urgent",
    "theatre",
    "theater",
}

REMOVE_AFTER = {
    "att at law",
    "attorney",
    "attorney-at-law",
    "contracto",
    "cpa",
    "crna",
    "d c p a",
    "d d s",
    "d p m",
    "d/b/a",
    "dba",
    "ddm",
    "dds",
    "ddspa",
    "ddspc",
    "dmd",
    "dmdpa",
    "do",
    "dpm",
    "dpmpa",
    "dvm",
    "esq",
    "et al",
    "hairdresser",
    "law",
    "m d",
    "md",
    "o d",
    "mda",
    "mdpc",
    "mdpllc",
    "od",
    "or",
    "p a",
    "pa",
    "pc",
    "pe",
    "phd",
    "pl",
    "pll",
    "psc",
    "pt",
    "realtor",
    "uber driver",
}

LEFT_STRIP = {"dr"}

RIGHT_STRIP = {"jr", "i", "ii", "iii", "iv", "v"}


def coerce_to_ascii(string: str) -> str:
    return string.encode("ascii", errors="ignore").decode("ascii")


def is_all_consonants(word: str) -> bool:
    consonants = "bcdfghjklmnpqrstvwxyz"

    return all(c in consonants for c in word)


def contains_invalid_punctuation(word: str) -> bool:
    return cutils.contains(word, "!/`~@#$%^&*()-+{[]}")


def remove_after(expr: pl.Expr, by_list: Sequence[str]) -> pl.Expr:
    for by in by_list:
        expr = expr.add(" ")
        expr = expr.str.split(f" {by} ").list.get(0)
        expr = expr.str.rstrip(" ")

    return expr


def right_strip(expr: pl.Expr, char_list: Sequence[str]) -> pl.Expr:
    for char in char_list:
        expr = expr.str.rstrip(char)

    return expr


def left_strip(expr: pl.Expr, char_list: Sequence[str]) -> pl.Expr:
    for char in char_list:
        expr = expr.str.lstrip(char)

    return expr


def has_person_name(names: list[str]) -> list[bool]:
    unused = [p for p in NLP.pipe_names if p != "ner"]
    results = []
    with tqdm.tqdm(total=len(names)) as pbar:
        for doc in NLP.pipe(names, disable=unused, n_process=-1):
            results.append(any(ent.label_ == "PERSON" for ent in doc.ents))
            pbar.update(1)

    return results


def remove_non_person_names(overwrite: bool = False):
    if not overwrite:
        return pl.scan_parquet(PPP_PATH / "ppp_clean.parquet")

    ppp = (
        pl.scan_parquet(PPP_PATH / "ppp_raw.parquet")
        .select(pl.exclude("^__index.*$"))
        .filter(~pl.col("borrowername").is_null())
        .with_columns(pl.col("borrowername").str.replace_all("_", " ", literal=True))
        .with_columns(
            pl.col(
                "borrowername",
                "race",
                "ethnicity",
                "borrowerstate",
            )
            .str.to_lowercase()
            .str.strip(" ")
            .str.replace(r"\s+", " ")
        )
        .filter(~pl.col("borrowername").is_in({"", " "}))
        .filter(~pl.col("borrowername").apply(contains_invalid_punctuation))
        .with_columns(pl.col("borrowerzip").str.split("-").list.get(0))
        .with_columns(
            race_ethnicity=pl.when(pl.col("ethnicity") == "hispanic or latino")
            .then("hispanic")
            .otherwise(pl.col("race"))
        )
        .filter(pl.col("race_ethnicity") != "unanswered")
        .with_columns(
            pl.when(pl.col("borrowerzip").str.lengths() < 5)
            .then(None)
            .otherwise(pl.col("borrowerzip"))
            .keep_name()
        )
        .unique()
        .filter(~pl.col("borrowername").str.contains("[0-9]"))
        .with_columns(
            pl.col("borrowername")
            .str.replace_all(".", "", literal=True)
            .pipe(remove_after, REMOVE_AFTER)
            .pipe(right_strip, RIGHT_STRIP)
            # .pipe(left_strip, LEFT_STRIP)
            .str.strip()
        )
        .filter(~pl.col("borrowername").str.contains("|".join(INVALID_NAMES_ANY)))
        .filter(
            ~pl.any(
                pl.lit(item).is_in(pl.col("borrowername").str.split(" "))
                for item in INVALID_NAMES_WORD
            )
        )
        .with_columns(
            name_arr=pl.col("borrowername").str.split(" "),
        )
        .with_columns(
            name_arr_no_first=pl.when(pl.col("name_arr").list.lengths() >= 2)
            .then(pl.col("name_arr").list.slice(1))
            .otherwise(pl.col("name_arr")),
        )
        .with_columns(
            fname_len=pl.col("name_arr").list.get(0).str.lengths(),
        )
        .with_columns(
            borrowername=pl.when(pl.col("fname_len") < 2)
            .then(pl.col("name_arr_no_first").list.join(" "))
            .otherwise(pl.col("borrowername")),
        )
        .drop("name_arr", "name_arr_no_first", "fname_len")
        .with_columns(
            name_arr=pl.col("borrowername").str.split(" "),
        )
        .with_columns(
            name_arr_no_last=pl.when(pl.col("name_arr").list.lengths() >= 2)
            .then(pl.col("name_arr").list.slice(0, -2))
            .otherwise(pl.col("name_arr")),
        )
        .with_columns(
            lname_len=pl.col("name_arr").list.get(-1).str.lengths(),
        )
        .with_columns(
            borrowername=pl.when(pl.col("lname_len") == 1)
            .then(pl.col("name_arr_no_last").list.join(" "))
            .otherwise(pl.col("borrowername")),
        )
        .drop("name_arr", "name_arr_no_last", "lname_len")
        .with_columns(pl.col("businesstype").str.to_lowercase())
        .filter(
            ~(
                (pl.lit("church").is_in(pl.col("borrowername").str.split(" ")))
                & (pl.col("businesstype").str.contains("non-profit"))
            )
        )
        .filter(~pl.col("borrowername").is_null())
        .filter(~pl.col("borrowername").is_in({"", " "}))
        .filter(~(pl.col("borrowername").str.n_chars() <= 2))
        .collect()
    )

    borrowernames = ppp.get_column("borrowername").to_list()
    results = has_person_name(borrowernames)

    ppp = (
        ppp.with_columns(pl.Series("spacy_has_person_name", results))
        .filter(
            ~(
                (pl.col("borrowername").str.split(" ").list.lengths() >= 4)
                & ~(pl.col("spacy_has_person_name"))
            )
        )
        .filter(
            ~(
                (~pl.col("spacy_has_person_name"))
                & (pl.col("businesstype") != "sole proprietorship")
            )
        )
        .with_columns(tmp=pl.col("borrowername").str.split(" "))
        .with_columns(
            first_name=pl.col("tmp").list.get(0), last_name=pl.col("tmp").list.get(-1)
        )
        .with_columns(
            last_name=pl.when(pl.col("tmp").list.lengths() == 1)
            .then(None)
            .otherwise(pl.col("last_name"))
        )
        .drop("tmp")
        .with_columns(
            pl.col("race_ethnicity").str.replace(
                "black or african american", "black", literal=True
            )
        )
        .filter(pl.col("race_ethnicity").is_in(["black", "white", "asian", "hispanic"]))
        .filter(~(pl.col("last_name").str.starts_with("-")))
        .filter(~(pl.col("last_name").apply(is_all_consonants)))
        .filter(~(pl.col("first_name").apply(is_all_consonants)))
        .drop(
            "spacy_has_person_name", "borrowername", "race", "ethnicity", "businesstype"
        )
        .rename({"borrowerstate": "state_abbrev", "borrowerzip": "zip"})
    )

    ppp.write_parquet(PPP_PATH / "ppp_clean.parquet")

    return ppp.lazy()


def create_baseline_file(
    df: pl.LazyFrame, outpath: Path, overwrite: bool = False
) -> pl.LazyFrame:
    if not overwrite:
        return pl.scan_parquet(outpath)

    zip_zcta = pl.scan_parquet(CW_PATH / "zip_zcta_cw_final.parquet")
    race_pct = pl.scan_parquet(CENSUS_PATH / "race_by_zcta.parquet")
    final = (
        df.filter(
            ~(
                (pl.col("race_ethnicity").is_null())
                | (pl.col("race_ethnicity") == "null")
            )
        )
        .filter(
            pl.col("dateapproved").str.to_date(format="%m/%d/%Y")
            < datetime.date(2021, 2, 24)
            # 2/24/2021 is day that rules were changed to prefer black borrowers
            # explicitly. Without this filter, black is the dominant group, throwing
            # off the "true" racial distribution of the US
        )
        .with_columns(
            pl.col("first_name").str.replace_all(f"[^{VALID_NAME_CHARS}]", ""),
            pl.col("last_name").str.replace_all(f"[^{VALID_NAME_CHARS}]", ""),
        )
        .with_columns(
            first_name_length=pl.col("first_name").str.lengths(),
            last_name_length=pl.col("last_name").str.lengths(),
        )
        .filter(
            ~((pl.col("first_name_length") == 1) | (pl.col("last_name_length") == 1))
        )
        .select(pl.exclude("first_name_length", "last_name_length"))
        # don't do any filtering on if the name is "too long". By visual inspection,
        # these seem to still be valid names
        .join(zip_zcta, on="zip", how="left")
        .join(race_pct, on="zcta", how="left")
        .collect()
    )
    final = final.with_columns(pl.Series(name="index", values=range(final.shape[0])))
    final.write_parquet(outpath)

    return final.lazy()


def main():
    ppp = remove_non_person_names(overwrite=False)
    create_baseline_file(ppp, outpath=FINAL_PATH / "ppp_test.parquet", overwrite=True)


if __name__ == "__main__":
    main()
