from Player import Player
import csv
import urllib.request

class FetchFootballData:
    def fetch(url):
        players = []
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
            reader = csv.reader(data.splitlines())
            # Skip the header row
            next(reader)
            for row in reader:
                # Extract data from the row
                playerId, name, team, testData, trainData, allData= row
                # Create a new Player object and add it to the list
                player = Player(name, team, playerId, testData, trainData, allData)
                players.append(player)
        return players
