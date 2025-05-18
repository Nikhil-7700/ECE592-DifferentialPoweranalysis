`timescale 1ns/1ps

module aes_bruteforce_tb;

// Inputs
reg clock;
reg resetn;
reg enc_dec; // Encrypt/Decrypt select
reg key_exp; // Round Key Expansion
reg start;
reg [127:0] key_in;
reg [127:0] text_in;

// Outputs
wire [127:0] text_out;
wire key_val;
wire text_val;
wire busy;

// Instantiate the AES module
aes128_table_ecb aes_inst (
    .resetn(resetn),
    .clock(clock),
    .enc_dec(enc_dec),
    .key_exp(key_exp),
    .start(start),
    .key_val(key_val),
    .text_val(text_val),
    .key_in(key_in),
    .text_in(text_in),
    .text_out(text_out),
    .busy(busy)
);

// Variables
reg [127:0] known_ciphertext;  // The correct ciphertext we are trying to match
reg match_found;
integer i;

// Clock generation
always #5 clock = ~clock;

initial begin
    // Initialize inputs
    clock = 0;
    resetn = 1;
    start = 0;
    enc_dec = 0;  // 0 for encryption
    key_exp = 0;  // Set to initiate key expansion
    key_in = 128'b0;
    text_in = 128'hB9D1C48E348FE771FA464A77A178FB07; // Plaintext
    known_ciphertext = 128'h95F8847369A8573D76AF987AB30A5DE2;  // Known ciphertext
    match_found = 0;

    // Reset the AES module
    #10 resetn = 0;
    #10 resetn = 1;
    
    // Start brute-force attack
    for (i = 0; i < 65536; i = i + 1) begin
        if (match_found) begin
            $display("Key found: %h", key_in);
            $finish;  // Stop simulation
        end
        
        // Generate key for this iteration
	key_in[7:0] = (i[0]) ? 8'h0f : 8'hf0;  // Byte 1
        key_in[15:8] = (i[1]) ? 8'h0e : 8'hf1; // Byte 2
        key_in[23:16] = (i[2]) ? 8'h0d : 8'hf2; // Byte 3
        key_in[31:24] = (i[3]) ? 8'hf3 : 8'h0c; // Byte 4
        key_in[39:32] = (i[4]) ? 8'h0b : 8'hf4; // Byte 5
        key_in[47:40] = (i[5]) ? 8'hf5 : 8'h0a; // Byte 6
        key_in[55:48] = (i[6]) ? 8'h09 : 8'hf6; // Byte 7
        key_in[63:56] = (i[7]) ? 8'hf7 : 8'h08; // Byte 8
        key_in[71:64] = (i[8]) ? 8'h07 : 8'hf8; // Byte 9
        key_in[79:72] = (i[9]) ? 8'hf9 : 8'h06; // Byte 10
        key_in[87:80] = (i[10]) ? 8'h05 : 8'hfa; // Byte 11
        key_in[95:88] = (i[11]) ? 8'hfb : 8'h04; // Byte 12
        key_in[103:96] = (i[12]) ? 8'h03 : 8'hfc; // Byte 13
        key_in[111:104] = (i[13]) ? 8'h02 : 8'hfd; // Byte 14
        key_in[119:112] = (i[14]) ? 8'hfe : 8'h01; // Byte 15
        key_in[127:120] = (i[15]) ? 8'hff : 8'h00; // Byte 16

        // Start encryption
        start = 1;
        #10 start = 0;

        // Wait for AES to finish
        wait (key_val == 1'b1 && text_val == 1'b1);

        // Check if ciphertext matches the known ciphertext
        if (text_out == known_ciphertext) begin
            match_found = 1;
        end

        // Wait a bit before the next iteration
        #10;
    end

    // If we reach here, no match was found
    $display("No matching key found.");
    $finish;
end

endmodule

