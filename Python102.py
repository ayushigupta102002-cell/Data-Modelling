places = ["Goa", "Andaman and Nicobar Islands", "Mansarovar Lake", "Trincomalee", "Cambridge University", "Switzerland","Abisko","Great Barrier Reef","Sahara Desert","Salar de Uyuni"]
print("Top ten places I want to visit are:")
for place in places:
    print(place)
print(len(places) == 10)
print(sum(places.count(place)for place in places) == len(places))
places1 = places.copy()
places1.insert(places1.index("Trincomalee"),"Kedarnath")
print("Before any international trips, I want to visit Kedarnath.")
places1.remove("Cambridge University")
print("Cambridge University is only for academic purposes and hence removed.")
places1.append("Antarctica")
print("I want to visit Antarctica as well.")
places1.pop(0)
print("I have already visited Goa hence removed it from the list.")
print("Post final revision, top ten places I want to visit are:")
for place in places1:
    print(place)
print(len(places1) == 10)
print(len(set(places1)) == len(places1)) # approach 2 to check for duplicates in the list. more efficent than the previous approach.