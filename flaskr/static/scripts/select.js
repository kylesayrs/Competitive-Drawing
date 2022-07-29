var socket = io();

document.getElementById("localGameButton").onclick = function(_event) {
    window.location.href = "/local_game";
    //socket.emit("select", {"type": "localgame"});
};

socket.on("myurl", () => {
    console.log("asdf")
})
