"""
Mapping of actual intents to 'readable' intents. For use when evaluating semantic similarity with generated intents for some datasets
"""

readable_intents = {
    "atis": {
        "atis_abbreviation": "abbreviation",
        "atis_aircraft": "aircraft",
        "atis_aircraft#atis_flight#atis_flight_no": "aircraft, flight, flight number",
        "atis_airfare": "airfare",
        "atis_airfare#atis_flight_time": "airfare, flight time",
        "atis_airline": "airline",
        "atis_airline#atis_flight_no": "airline, flight number",
        "atis_airport": "airport",
        "atis_capacity": "capacity",
        "atis_cheapest": "cheapest",
        "atis_city": "city",
        "atis_distance": "distance",
        "atis_flight": "flight",
        "atis_flight#atis_airfare": "flight, airfare",
        "atis_flight_no": "flight number",
        "atis_flight_time": "flight time",
        "atis_ground_fare": "ground fare",
        "atis_ground_service": "ground service",
        "atis_ground_service#atis_ground_fare": "ground service, ground fare",
        "atis_meal": "meal",
        "atis_quantity": "quantity",
        "atis_restriction": "restriction"
    },
    "personal_assistant": {
            "addcontact": "add contact",
            "createoradd": "create or add",
            "hue_lightchange": "hue_light change",
            "hue_lightdim": "hue_light dim",
            "hue_lightoff": "hue_light off",
            "hue_lighton": "hue_light on",
            "hue_lightup": "hue_light up",
            "querycontact": "query contact",
            "sendemail": "send email"
    },
    "twacs": {
        "BOOKFLT": "book flight",
        "CHGFLT": "change flight",
        "CHECKIN": "check in",
        "CUSTSUPP": "customer support",
        "INFLIGHTENT": "flight entertainment",
        "INFLIGHTFACS": "flight facilities",
        "FLIGHTSTAFF": "flight staff",
        "NEWFEAT": "request feature",
        "TERMFACS": "terminal facilities",
        "TERMOPS": "terminal operations",
    }
}            
    