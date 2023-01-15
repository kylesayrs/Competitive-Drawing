import { GameBase } from "/static/scripts/game/base.js";


export class OnlineGame extends GameBase {
    constructor(gameConfig, debug=false) {
        super(2, gameConfig, debug)

        // Player specific variables
        this.playerTargetIndex = null
        this.myTurn = false

        // Hook with drawing board
        /*
        this.drawingBoard.afterMouseEnd = async () => {
            if (this.myTurn) {
                this.serverInferImage()
            }
        }
        */

        // Initialize components
        this.drawingBoard.enabled = false
        this.distanceIndicator.emptyDistance()
    }


    onStartGame(data) {
        super.onStartGame(data)

        // assign target index
        this.playerTargetIndex = data["targetIndices"][this.playerId]
    }


    async onStartTurn(data) {
        super.onStartTurn(data)

        if (data["turn"] == this.playerId) {
            this.myTurn = true
            this.drawingBoard.enabled = true
            this.distanceIndicator.resetDistance()

            Toastify({
                text: "Your turn",
                duration: 3000,
                className: "info",
                gravity: "toastify-top",
                style: {
                    background: "linear-gradient(to right, #2ec1cc, #0074d9)",
                }
            }).showToast()
        } else {
            this.myTurn = false
            this.drawingBoard.enabled = false
            this.distanceIndicator.emptyDistance()
        }
    }
}
