def _parse_metadata(self, message):
        

        Returns:
            Legobot.Metadata
        "

        metadata = Metadata(source=self.actor_urn).__dict__
        if  in message[]:
            metadata[] = message[][][]
        else:
            metadata[] = None
        if  in message[]:
            metadata[] = message[][]
        else:
            metadata[] = None
        metadata[] = metadata[]
        metadata[] = metadata[]

        metadata[] = 

        return metadata