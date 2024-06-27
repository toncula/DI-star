import openai
import os
from openai import OpenAI
import time

os.environ["HTTP_PROXY"] = "http://127.0.0.1:23457"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:23457"
openai.api_key = os.environ.get("OPENAI_API_KEY")

def llm_select(actions,LLM_input,bo,sum = ""):
    client = OpenAI(api_key=openai.api_key)
    system_message = f"""
    You are an AI trained in analyzing and summarizing StarCraft II games.You understand the nuances and strategies of the zerg race.
    Based on human input and selected action output of rl_model, we want you to analyze the game and give the perfect match action from given actions to human_input.

    Knowledge about actions:
    1.'train_sth': trying to train given unit
    2.'Research_sth':trying to reserch given tech
    3.'bulid_sth':trying to bulid given buliding, at location tu
    4.'Attack_pt' or 'Attack_unit' attack given unit or point
    5.'Smart_pt' or 'Smart_unit' usually move to given point or unit, attack if meet enemy
    6. 'effect' usually using unit's ability
    7. 'morph' usually change unit's form. Burrow/unBurrow for zerg unit.
    Knowledge about game:
    1.you are in a zerg vs zerg match, with starting location:{bo}
    Kairos Junction Map Description
    General Overview
    Map Size: 160x160 units
    Type: 2-player map
    Symmetry: Diagonal symmetry from top-left to bottom-right
    Main Bases
    Top-left Main Base (Position A): Centered approximately at (30, 130).
    Features a main base area with ramps leading down to lower ground.
    Bottom-right Main Base (Position B): Centered approximately at (130, 30).
    Similarly structured with ramps leading down to the lower ground.
    Natural Expansions
    Top-left Natural Expansion: Located at (50, 110), directly outside the main base.
    Bottom-right Natural Expansion: Located at (110, 50), directly outside the main base.
    Third Base Locations
    Top-left Third Base:
    Can be taken at (70, 90), which is closer and safer.
    Alternatively, at (30, 70) which is more exposed but offers a forward position.
    Bottom-right Third Base:
    Can be taken at (90, 70), which is closer and safer.
    Alternatively, at (130, 90) which is more exposed but offers a forward position.
    Other Expansions
    Top-left Side Expansion: Located at (20, 50).
    Bottom-right Side Expansion: Located at (140, 110).
    Central Expansions:
    Two expansions in the center at (80, 80) and (80, 100).
    Routes and Choke Points
    Primary Attack Routes:
    Top-left to Bottom-right: Routes start from (30, 130) and (130, 30) and converge towards the central area.
    Secondary Routes: Lead through the mid-map expansions, creating multiple pathways for strategic maneuvering.
    Choke Points:
    Key choke points exist near the natural expansions at (50, 110) and (110, 50).
    Additional chokes are found near the third base locations and central expansions, adding layers of defensive and offensive options.
    Terrain Features
    High Ground Areas: Near the main bases, particularly around the (30, 130) and (130, 30) regions.
    Rocks and Destructible Debris: Present near central expansions at (80, 80) and (80, 100), controlling access and requiring tactical decisions for clearing.
    Map Control and Vision
    Watchtowers:
    One watchtower centrally located at (80, 80), providing significant vision over the middle of the map.
    Vision Blockers:
    Various small terrain features and destructible rocks create vision obstructions, especially around the natural expansions and central areas.

    output format:a certain number repersent the selected action's index in action dictionary.
    output example:15
    """
    MODEL = "gpt-4o"
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": f"current state summarize:{sum},action:{actions},human_input:{LLM_input},output format:please output a single number repersent the selected action's index in action dictionary,do not output any explaination about that",
                },
            ],
        )
        try:
            ret = completion.choices[0].message.content
            ret = int(ret)
            if ret < 0 or ret > 19:
                ret = 0
        except:
            ret = 0
            print(completion.choices[0].message.content)
    except:
        print("Warning: API connection error occurred. Sending empty command.")
        ret = 0

    return ret


def llm_summarize(player_common, visible_units,retries=3):
    client = OpenAI(api_key=openai.api_key)
    MODEL = "gpt-4o"
    system_message = f"""You are an AI trained in analyzing and summarizing StarCraft II games.You understand the nuances and strategies of the zerg race.
    based on current observation about player_common and s2clientprotocol.raw_pb2.Unit format units data,
    please summarize current game situation.
    Kairos Junction Map Description
    General Overview
    Map Size: 160x160 units
    Type: 2-player map
    Symmetry: Diagonal symmetry from top-left to bottom-right
    Main Bases
    Top-left Main Base (Position A): Centered approximately at (30, 130).
    Features a main base area with ramps leading down to lower ground.
    Bottom-right Main Base (Position B): Centered approximately at (130, 30).
    Similarly structured with ramps leading down to the lower ground.
    Natural Expansions
    Top-left Natural Expansion: Located at (50, 110), directly outside the main base.
    Bottom-right Natural Expansion: Located at (110, 50), directly outside the main base.
    Third Base Locations
    Top-left Third Base:
    Can be taken at (70, 90), which is closer and safer.
    Alternatively, at (30, 70) which is more exposed but offers a forward position.
    Bottom-right Third Base:
    Can be taken at (90, 70), which is closer and safer.
    Alternatively, at (130, 90) which is more exposed but offers a forward position.
    Other Expansions
    Top-left Side Expansion: Located at (20, 50).
    Bottom-right Side Expansion: Located at (140, 110).
    Central Expansions:
    Two expansions in the center at (80, 80) and (80, 100).
    Routes and Choke Points
    Primary Attack Routes:
    Top-left to Bottom-right: Routes start from (30, 130) and (130, 30) and converge towards the central area.
    Secondary Routes: Lead through the mid-map expansions, creating multiple pathways for strategic maneuvering.
    Choke Points:
    Key choke points exist near the natural expansions at (50, 110) and (110, 50).
    Additional chokes are found near the third base locations and central expansions, adding layers of defensive and offensive options.
    Terrain Features
    High Ground Areas: Near the main bases, particularly around the (30, 130) and (130, 30) regions.
    Rocks and Destructible Debris: Present near central expansions at (80, 80) and (80, 100), controlling access and requiring tactical decisions for clearing.
    Map Control and Vision
    Watchtowers:
    One watchtower centrally located at (80, 80), providing significant vision over the middle of the map.
    Vision Blockers:
    Various small terrain features and destructible rocks create vision obstructions, especially around the natural expansions and central areas.

    the dictionary of zerg's unit:
    Baneling = 9
    BanelingBurrowed = 115
    BanelingCocoon = 8
    BanelingNest = 96
    BroodLord = 114
    BroodLordCocoon = 113
    Broodling = 289
    BroodlingEscort = 143
    Changeling = 12
    ChangelingMarine = 15
    ChangelingMarineShield = 14
    ChangelingZealot = 13
    ChangelingZergling = 17
    ChangelingZerglingWings = 16
    Cocoon = 103
    Corruptor = 112
    CreepTumor = 87
    CreepTumorBurrowed = 137
    CreepTumorQueen = 138
    Drone = 104
    DroneBurrowed = 116
    EvolutionChamber = 90
    Extractor = 88
    ExtractorRich = 1956
    GreaterSpire = 102
    Hatchery = 86
    Hive = 101
    Hydralisk = 107
    HydraliskBurrowed = 117
    HydraliskDen = 91
    InfestationPit = 94
    InfestedTerran = 7
    InfestedTerranBurrowed = 120
    InfestedTerranCocoon = 150
    Infestor = 111
    InfestorBurrowed = 127
    Lair = 100
    Larva = 151
    Locust = 489
    LocustFlying = 693
    Lurker = 502
    LurkerBurrowed = 503
    LurkerDen = 504
    LurkerCocoon = 501
    Mutalisk = 108
    NydusCanal = 142
    NydusNetwork = 95
    Overlord = 106
    OverlordTransport = 893
    OverlordTransportCocoon = 892
    Overseer = 129
    OverseerCocoon = 128
    OverseerOversightMode = 1912
    ParasiticBombDummy = 824
    Queen = 126
    QueenBurrowed = 125
    Ravager = 688
    RavagerBurrowed = 690
    RavagerCocoon = 687
    Roach = 110
    RoachBurrowed = 118
    RoachWarren = 97
    SpawningPool = 89
    SpineCrawler = 98
    SpineCrawlerUprooted = 139
    Spire = 92
    SporeCrawler = 99
    SporeCrawlerUprooted = 140
    SwarmHost = 494
    SwarmHostBurrowed = 493
    Ultralisk = 109
    UltraliskBurrowed = 131
    UltraliskCavern = 93
    Viper = 499
    Zergling = 105
    ZerglingBurrowed = 119

    please describe and summarize the current situation as detailed as possible.

    """
    attempt = 0
    while attempt < retries:
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": f"player_common:{player_common}, units:{visible_units}",
                    },
                ],
            )
            ret = completion.choices[0].message.content
            return ret
        except Exception as e:
            attempt += 1
            print(f"Warning: API connection error occurred. Retrying {attempt}/{retries}...")
    print("Error: API connection failed after multiple attempts.")
    return ""
