#ifndef IO_HH
#define IO_HH

#include <cstdlib>
#include <string>
#include <stdio.h>

/** Function and classes for opening files, compressed files, or
 * process pipes transparently. 
 *
 * Example:
 * \include ex_stream.cc
 */
namespace io {
  typedef enum {LINEAR, REOPENABLE, SEEKABLE} SeekType;

  /** A general stream class.  A stream can be a file or a pipe. */
  struct Stream {
    
    /** Create an empty stream. */
    Stream();

    /** Create and open a stream (see open() for details). */
    Stream(std::string file_name, std::string mode, SeekType seek_type=LINEAR, 
	   bool allow_close = true);

    /** Destroy the stream and close the possible open file if \ref
     * close_allowed is \c true. */
    ~Stream();

    /** Open a stream.  If the file name ends with ".gz" a pipe
     * through gzip is created.  If the file name ends with "|", an
     * input pipe is created.  In that case, the mode must be "r".  If
     * the file name starts with "|", an output pipe is created.  If
     * the stream was open already, close the previous stream before
     * opening a new one.
     *
     * \param file_name = the file or pipe to open 
     * \param mode = the mode for the file.  See the man page of
     * fopen() for more details.  Possible values are "r", "r+", "w",
     * "w+", "a", "a+", but only "r" and "w" are supported for pipes
     * and gzipped files.
     * \param allow_close = should the close() actually close the file
     */
    void open(std::string file_name, std::string mode, 
	      SeekType seek_type=LINEAR, bool allow_close = true);

    /** Close the file, but only if \ref close_allowed is \c true. */
    void close();

    FILE *file; //!< The handle for the file or pipe
    bool is_pipe; //!< Is the opened file pipe that must be closed with pclose
    bool close_allowed; //!< Should the close() function close the file

    static bool verbose; // Print stuff

  private:
    
    // Do not allow copying streams!
    Stream(const Stream &stream) { abort(); }
    const Stream &operator=(const Stream &stream) { abort(); return(stream); }

  };
};

#endif /* IO_HH */
