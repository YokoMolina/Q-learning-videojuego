#ifndef __ALE_C_WRAPPER_H__
#define __ALE_C_WRAPPER_H__

#include <ale_interface.hpp>

#ifdef _WIN32
#  define SYMBOL_EXPORT __declspec(dllexport)
#endif
#ifndef SYMBOL_EXPORT
#  define SYMBOL_EXPORT
#endif

extern "C" {
  // Declares int rgb_palette[256]
  #include "atari_ntsc_rgb_palette.h"
  SYMBOL_EXPORT ALEInterface *ALE_new() {return new ALEInterface();}
  SYMBOL_EXPORT void ALE_del(ALEInterface *ale){delete ale;}
  SYMBOL_EXPORT const char *getString(ALEInterface *ale, const char *key){return ale->getString(key).c_str();}
  SYMBOL_EXPORT int getInt(ALEInterface *ale,const char *key) {return ale->getInt(key);}
  SYMBOL_EXPORT bool getBool(ALEInterface *ale,const char *key){return ale->getBool(key);}
  SYMBOL_EXPORT float getFloat(ALEInterface *ale,const char *key){return ale->getFloat(key);}
  SYMBOL_EXPORT void setString(ALEInterface *ale,const char *key,const char *value){ale->setString(key,value);}
  SYMBOL_EXPORT void setInt(ALEInterface *ale,const char *key,int value){ale->setInt(key,value);}
  SYMBOL_EXPORT void setBool(ALEInterface *ale,const char *key,bool value){ale->setBool(key,value);}
  SYMBOL_EXPORT void setFloat(ALEInterface *ale,const char *key,float value){ale->setFloat(key,value);}
  SYMBOL_EXPORT bool loadROM(ALEInterface *ale,const char *rom,size_t size,const char *name){return ale->loadROM(std::string(rom, size), name);}
  SYMBOL_EXPORT int act(ALEInterface *ale,int action){return ale->act((Action)action);}
  SYMBOL_EXPORT bool game_over(ALEInterface *ale){return ale->game_over();}
  SYMBOL_EXPORT void reset_game(ALEInterface *ale){ale->reset_game();}
  SYMBOL_EXPORT void getAvailableModes(ALEInterface *ale,int *availableModes) {
    ModeVect modes_vect = ale->getAvailableModes();
    for(unsigned int i = 0; i < ale->getAvailableModes().size(); i++){
      availableModes[i] = modes_vect[i];
    }
  }
  SYMBOL_EXPORT int getAvailableModesSize(ALEInterface *ale) {return ale->getAvailableModes().size();}
  SYMBOL_EXPORT void setMode(ALEInterface *ale, int mode) {ale->setMode(mode);}
  SYMBOL_EXPORT void getAvailableDifficulties(ALEInterface *ale,int *availableDifficulties) {
    DifficultyVect difficulties_vect = ale->getAvailableDifficulties();
    for(unsigned int i = 0; i < ale->getAvailableDifficulties().size(); i++){
      availableDifficulties[i] = difficulties_vect[i];
    }
  }
  SYMBOL_EXPORT int getAvailableDifficultiesSize(ALEInterface *ale) {return ale->getAvailableDifficulties().size();}
  SYMBOL_EXPORT void setDifficulty(ALEInterface *ale, int difficulty) {ale->setDifficulty(difficulty);}
  SYMBOL_EXPORT void getLegalActionSet(ALEInterface *ale,int *actions){
    ActionVect action_vect = ale->getLegalActionSet();
    for(unsigned int i = 0; i < ale->getLegalActionSet().size(); i++){
      actions[i] = action_vect[i];
    }
  }
  SYMBOL_EXPORT int getLegalActionSize(ALEInterface *ale){return ale->getLegalActionSet().size();}
  SYMBOL_EXPORT void getMinimalActionSet(ALEInterface *ale,int *actions){
    ActionVect action_vect = ale->getMinimalActionSet();
    for(unsigned int i = 0;i < ale->getMinimalActionSet().size();i++){
      actions[i] = action_vect[i];
    }
  }
  SYMBOL_EXPORT int getMinimalActionSize(ALEInterface *ale){return ale->getMinimalActionSet().size();}
  SYMBOL_EXPORT int getFrameNumber(ALEInterface *ale){return ale->getFrameNumber();}
  SYMBOL_EXPORT int lives(ALEInterface *ale){return ale->lives();}
  SYMBOL_EXPORT int getEpisodeFrameNumber(ALEInterface *ale){return ale->getEpisodeFrameNumber();}
  SYMBOL_EXPORT void getScreen(ALEInterface *ale,unsigned char *screen_data){
    int w = ale->getScreen().width();
    int h = ale->getScreen().height();
    pixel_t *ale_screen_data = (pixel_t *)ale->getScreen().getArray();
    memcpy(screen_data,ale_screen_data,w*h*sizeof(pixel_t));
  }
  SYMBOL_EXPORT void getRAM(ALEInterface *ale,unsigned char *ram){
    unsigned char *ale_ram = ale->getRAM().array();
    int size = ale->getRAM().size();
    memcpy(ram,ale_ram,size*sizeof(unsigned char));
  }
  SYMBOL_EXPORT int getRAMSize(ALEInterface *ale){return ale->getRAM().size();}
  SYMBOL_EXPORT int getScreenWidth(ALEInterface *ale){return ale->getScreen().width();}
  SYMBOL_EXPORT int getScreenHeight(ALEInterface *ale){return ale->getScreen().height();}

  SYMBOL_EXPORT void getScreenRGB(ALEInterface *ale, unsigned char *output_buffer){
    size_t w = ale->getScreen().width();
    size_t h = ale->getScreen().height();
    size_t screen_size = w*h;
    pixel_t *ale_screen_data = ale->getScreen().getArray();

    ale->theOSystem->colourPalette().applyPaletteRGB(output_buffer, ale_screen_data, screen_size );
  }


  SYMBOL_EXPORT void getScreenRGB2(ALEInterface *ale, unsigned char *output_buffer){
    size_t w = ale->getScreen().width();
    size_t h = ale->getScreen().height();
    size_t screen_size = w*h;
    pixel_t *ale_screen_data = ale->getScreen().getArray();

    int j = 0;
    for(int i = 0;i < screen_size;i++){
        unsigned int zrgb = rgb_palette[ale_screen_data[i]];
        output_buffer[j++] = (zrgb>>16)&0xff;
        output_buffer[j++] = (zrgb>>8)&0xff;
        output_buffer[j++] = (zrgb>>0)&0xff;
    }
  }


  SYMBOL_EXPORT void getScreenGrayscale(ALEInterface *ale, unsigned char *output_buffer){
    size_t w = ale->getScreen().width();
    size_t h = ale->getScreen().height();
    size_t screen_size = w*h;
    pixel_t *ale_screen_data = ale->getScreen().getArray();

    ale->theOSystem->colourPalette().applyPaletteGrayscale(output_buffer, ale_screen_data, screen_size);
  }

  SYMBOL_EXPORT void saveState(ALEInterface *ale){ale->saveState();}
  SYMBOL_EXPORT void loadState(ALEInterface *ale){ale->loadState();}
  SYMBOL_EXPORT ALEState* cloneState(ALEInterface *ale){return new ALEState(ale->cloneState());}
  SYMBOL_EXPORT void restoreState(ALEInterface *ale, ALEState* state){ale->restoreState(*state);}
  SYMBOL_EXPORT ALEState* cloneSystemState(ALEInterface *ale){return new ALEState(ale->cloneSystemState());}
  SYMBOL_EXPORT void restoreSystemState(ALEInterface *ale, ALEState* state){ale->restoreSystemState(*state);}
  SYMBOL_EXPORT void deleteState(ALEState* state){delete state;}
  //Kojoley   SYMBOL_EXPORT void saveScreenPNG(ALEInterface *ale,const char *filename){ale->saveScreenPNG(filename);}

  // Encodes the state as a raw bytestream. This may have multiple '\0' characters
  // and thus should not be treated as a C string. Use encodeStateLen to find the length
  // of the buffer to pass in, or it will be overrun as this simply memcpys bytes into the buffer.
  SYMBOL_EXPORT void encodeState(ALEState *state, char *buf, int buf_len);
  SYMBOL_EXPORT int encodeStateLen(ALEState *state);
  SYMBOL_EXPORT ALEState *decodeState(const char *serialized, int len);

  // 0: Info, 1: Warning, 2: Error
  SYMBOL_EXPORT void setLoggerMode(int mode) { ale::Logger::setMode(ale::Logger::mode(mode)); }
}

#endif
